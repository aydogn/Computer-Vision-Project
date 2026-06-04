"""
Ablation study training script for SAM ViT-B decoder fine-tuning.

Supported experiments (controlled via CLI flags):
  A1 — BCE only loss          : --bce-weight 1.0 --dice-weight 0.0
  A2 — Dice only loss         : --bce-weight 0.0 --dice-weight 1.0
  A3 — No augmentation        : --no-aug
  A4 — High learning rate     : --lr 1e-3
  A5 — Unfreeze prompt encoder: --unfreeze-prompt-encoder
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import CustomSAMDataset, get_val_test_transforms


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
        bce_loss = self.bce(logits, targets) if self.bce_weight > 0 else 0.0

        probs = torch.sigmoid(logits)
        smooth = 1e-6
        intersection = (probs * targets).sum(dim=(2, 3))
        dice_loss = 1 - (2 * intersection + smooth) / (
            probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth
        )
        dice_loss = dice_loss.mean() if self.dice_weight > 0 else 0.0

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
# Bounding box prompt from masks
# ---------------------------------------------------------------------------

def masks_to_boxes(masks):
    """masks: [B, 1, H, W] float tensor -> [B, 4] (x1, y1, x2, y2)"""
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
# Transforms
# ---------------------------------------------------------------------------

def get_train_transforms_with_aug():
    return A.Compose([
        A.LongestMaxSize(max_size=1024, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(min_height=1024, min_width=1024,
                      border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_train_transforms_no_aug():
    """Resize + pad + normalize only — no augmentation."""
    return A.Compose([
        A.LongestMaxSize(max_size=1024, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(min_height=1024, min_width=1024,
                      border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------

def run_epoch(sam, dataloader, criterion, optimizer, device, is_train, unfreeze_prompt):
    sam.mask_decoder.train() if is_train else sam.mask_decoder.eval()
    if unfreeze_prompt:
        sam.prompt_encoder.train() if is_train else sam.prompt_encoder.eval()

    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0
    n_batches = len(dataloader)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in dataloader:
            images = batch["image"].to(device)
            masks  = batch["mask"].to(device)
            B = images.shape[0]

            with torch.no_grad():
                image_embeddings = sam.image_encoder(images)

            batch_logits = []
            boxes_all = masks_to_boxes(masks)

            for i in range(B):
                emb = image_embeddings[i].unsqueeze(0)
                box = boxes_all[i].unsqueeze(0).unsqueeze(0)

                sparse_emb, dense_emb = sam.prompt_encoder(
                    points=None, boxes=box, masks=None,
                )
                low_res_logit, _ = sam.mask_decoder(
                    image_embeddings=emb,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
                batch_logits.append(low_res_logit)

            low_res_logits = torch.cat(batch_logits, dim=0)
            logits = F.interpolate(low_res_logits, size=(1024, 1024),
                                   mode="bilinear", align_corners=False)

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
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from segment_anything import sam_model_registry
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Run: pip install -r requirements.txt")

    print(f"\n{'='*60}")
    print(f"Experiment : {args.experiment_name}")
    print(f"Loss       : BCE={args.bce_weight}  Dice={args.dice_weight}")
    print(f"LR         : {args.lr}")
    print(f"Augment    : {not args.no_aug}")
    print(f"Prompt enc : {'trainable' if args.unfreeze_prompt_encoder else 'frozen'}")
    print(f"Output     : {args.output_dir}")
    print(f"{'='*60}\n")

    sam = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    sam.to(device)

    for param in sam.image_encoder.parameters():
        param.requires_grad = False

    for param in sam.prompt_encoder.parameters():
        param.requires_grad = args.unfreeze_prompt_encoder

    trainable = sum(p.numel() for p in sam.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in sam.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    train_transform = (get_train_transforms_no_aug() if args.no_aug
                       else get_train_transforms_with_aug())

    train_dataset = CustomSAMDataset(args.data_root, split="train",      transform=train_transform)
    val_dataset   = CustomSAMDataset(args.data_root, split="validation", transform=get_val_test_transforms())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    criterion = BCEDiceLoss(bce_weight=args.bce_weight, dice_weight=args.dice_weight)

    trainable_params = filter(lambda p: p.requires_grad, sam.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))

    best_val_dice  = 0.0
    best_ckpt_path = os.path.join(args.output_dir, "sam_finetuned_best.pth")

    print(f"Training — {args.epochs} epochs\n{'='*60}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_dice, train_iou = run_epoch(
            sam, train_loader, criterion, optimizer, device,
            is_train=True, unfreeze_prompt=args.unfreeze_prompt_encoder
        )
        val_loss, val_dice, val_iou = run_epoch(
            sam, val_loader, criterion, optimizer, device,
            is_train=False, unfreeze_prompt=args.unfreeze_prompt_encoder
        )
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"TrainLoss: {train_loss:.4f}  Dice: {train_dice:.4f}  IoU: {train_iou:.4f} | "
            f"ValLoss: {val_loss:.4f}  Dice: {val_dice:.4f}  IoU: {val_iou:.4f} | "
            f"{elapsed:.1f}s"
        )

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Dice", {"train": train_dice, "val": val_dice}, epoch)
        writer.add_scalars("IoU",  {"train": train_iou,  "val": val_iou},  epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(sam.state_dict(), best_ckpt_path)
            print(f"  -> Best model saved (Val Dice: {best_val_dice:.4f})")

    torch.save(sam.state_dict(), os.path.join(args.output_dir, "sam_finetuned_last.pth"))
    writer.close()

    print(f"\nTraining complete.")
    print(f"  Best checkpoint : {best_ckpt_path}  (Val Dice: {best_val_dice:.4f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SAM Ablation Study Training")
    parser.add_argument("--experiment-name", default="ablation",         help="Name tag for this run")
    parser.add_argument("--data-root",        default="kvasir-seg",       help="Kvasir-SEG root")
    parser.add_argument("--checkpoint",       default="checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--output-dir",       default="results/ablation/run")
    parser.add_argument("--epochs",           type=int,   default=20)
    parser.add_argument("--batch-size",       type=int,   default=2)
    parser.add_argument("--lr",               type=float, default=1e-4)
    parser.add_argument("--weight-decay",     type=float, default=1e-4)
    parser.add_argument("--bce-weight",       type=float, default=0.5,   help="Weight for BCE loss")
    parser.add_argument("--dice-weight",      type=float, default=0.5,   help="Weight for Dice loss")
    parser.add_argument("--no-aug",           action="store_true",        help="Disable training augmentation")
    parser.add_argument("--unfreeze-prompt-encoder", action="store_true", help="Also train the prompt encoder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
