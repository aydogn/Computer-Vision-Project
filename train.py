"""
Asama 3: SAM ViT-B Decoder Fine-tuning
- Image Encoder (ViT) tamamen dondurulur
- Sadece Mask Decoder egitilir
- Loss: BCE + Dice
- Optimizer: AdamW
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataloaders


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        smooth = 1e-6
        intersection = (probs * targets).sum(dim=(2, 3))
        dice_loss = 1 - (2 * intersection + smooth) / (
            probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth
        )
        dice_loss = dice_loss.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_batch_metrics(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) > threshold).float()
    smooth = 1e-6
    intersection = (preds * targets).sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (
        preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth
    )
    union = (preds + targets - preds * targets).sum(dim=(2, 3))
    iou = (intersection + smooth) / (union + smooth)
    return dice.mean().item(), iou.mean().item()


# ---------------------------------------------------------------------------
# Prompt: ground-truth bounding box (egitim icin)
# ---------------------------------------------------------------------------

def masks_to_boxes(masks):
    """
    masks: [B, 1, H, W] float tensor (0/1)
    Returns: [B, 4] float tensor (x1, y1, x2, y2) normalised to [0, 1024]
    """
    B = masks.shape[0]
    boxes = torch.zeros(B, 4, device=masks.device)
    for i in range(B):
        m = masks[i, 0]
        ys, xs = torch.where(m > 0.5)
        if len(xs) == 0:
            boxes[i] = torch.tensor([0, 0, 1024, 1024], dtype=torch.float32, device=masks.device)
        else:
            boxes[i] = torch.tensor(
                [xs.min(), ys.min(), xs.max(), ys.max()], dtype=torch.float32, device=masks.device
            )
    return boxes


# ---------------------------------------------------------------------------
# Tek epoch egitim / validasyon
# ---------------------------------------------------------------------------

def run_epoch(sam, dataloader, criterion, optimizer, device, is_train):
    if is_train:
        sam.mask_decoder.train()
    else:
        sam.mask_decoder.eval()

    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0
    n_batches = len(dataloader)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in dataloader:
            images = batch["image"].to(device)   # [B, 3, 1024, 1024]
            masks  = batch["mask"].to(device)    # [B, 1, 1024, 1024]

            # --- Image Encoding (no grad, encoder frozen) ---
            with torch.no_grad():
                image_embeddings = sam.image_encoder(images)

            # --- Prompt Encoding (bounding box) ---
            boxes = masks_to_boxes(masks)        # [B, 4]
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=None,
                boxes=boxes.unsqueeze(1),        # [B, 1, 4]
                masks=None,
            )

            # --- Mask Decoding ---
            low_res_logits, _ = sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            # Upsample to 1024x1024
            logits = F.interpolate(
                low_res_logits, size=(1024, 1024), mode="bilinear", align_corners=False
            )

            loss = criterion(logits, masks)
            dice, iou = compute_batch_metrics(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_dice += dice
            total_iou  += iou

    return total_loss / n_batches, total_dice / n_batches, total_iou / n_batches


# ---------------------------------------------------------------------------
# Ana egitim dongusu
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    try:
        from segment_anything import sam_model_registry
    except ModuleNotFoundError:
        raise ModuleNotFoundError("pip install -r requirements.txt ile segment-anything yukleyin.")

    print(f"SAM checkpoint yukleniyor: {args.checkpoint}")
    sam = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    sam.to(device)

    # Encoder + Prompt Encoder'i dondur
    for param in sam.image_encoder.parameters():
        param.requires_grad = False
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in sam.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in sam.parameters())
    print(f"Egitilecek parametre: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        root_dir=args.data_root, batch_size=args.batch_size
    )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # Loss & Optimizer
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, sam.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # TensorBoard
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))

    best_val_dice = 0.0
    best_ckpt_path = os.path.join(args.output_dir, "sam_finetuned_best.pth")

    print(f"\nEgitim basliyor — {args.epochs} epoch\n{'='*55}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_dice, train_iou = run_epoch(
            sam, train_loader, criterion, optimizer, device, is_train=True
        )
        val_loss, val_dice, val_iou = run_epoch(
            sam, val_loader, criterion, optimizer, device, is_train=False
        )
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"TrainLoss: {train_loss:.4f}  Dice: {train_dice:.4f}  IoU: {train_iou:.4f} | "
            f"ValLoss: {val_loss:.4f}  Dice: {val_dice:.4f}  IoU: {val_iou:.4f} | "
            f"{elapsed:.1f}s"
        )

        writer.add_scalars("Loss",      {"train": train_loss, "val": val_loss},       epoch)
        writer.add_scalars("Dice",      {"train": train_dice, "val": val_dice},       epoch)
        writer.add_scalars("IoU",       {"train": train_iou,  "val": val_iou},        epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # En iyi modeli kaydet
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(sam.state_dict(), best_ckpt_path)
            print(f"  -> Yeni en iyi model kaydedildi (Val Dice: {best_val_dice:.4f})")

    # Son epoch checkpointi
    last_ckpt_path = os.path.join(args.output_dir, "sam_finetuned_last.pth")
    torch.save(sam.state_dict(), last_ckpt_path)
    writer.close()

    print(f"\nEgitim tamamlandi.")
    print(f"  En iyi checkpoint : {best_ckpt_path}  (Val Dice: {best_val_dice:.4f})")
    print(f"  Son checkpoint    : {last_ckpt_path}")
    print(f"  TensorBoard loglar: {os.path.join(args.output_dir, 'tensorboard')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SAM Decoder Fine-tuning on Kvasir-SEG")
    parser.add_argument("--data-root",    default="kvasir-seg",                           help="Kvasir-SEG root")
    parser.add_argument("--checkpoint",   default="checkpoints/sam_vit_b_01ec64.pth",     help="SAM ViT-B checkpoint")
    parser.add_argument("--output-dir",   default="results/finetune",                     help="Cikti dizini")
    parser.add_argument("--epochs",       type=int,   default=20,    help="Epoch sayisi")
    parser.add_argument("--batch-size",   type=int,   default=2,     help="Batch size (VRAM'e gore)")
    parser.add_argument("--lr",           type=float, default=1e-4,  help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,  help="AdamW weight decay")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
