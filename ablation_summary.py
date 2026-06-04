"""
Evaluates all ablation checkpoints on the test set and prints a comparison table.

Usage:
    python ablation_summary.py

Expects the following directory structure (created by run_ablations.sh):
    results/ablation/A1_bce_only/sam_finetuned_best.pth
    results/ablation/A2_dice_only/sam_finetuned_best.pth
    results/ablation/A3_no_aug/sam_finetuned_best.pth
    results/ablation/A4_high_lr/sam_finetuned_best.pth
    results/ablation/A5_prompt_encoder/sam_finetuned_best.pth

The baseline result (already evaluated) is read from:
    results/finetuned_eval/finetuned_summary.txt
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

BASELINE = {
    "name": "Baseline (BCE+Dice, lr=1e-4, aug, decoder only)",
    "mean_dice": 0.937444,
    "mean_iou":  0.892464,
}

ABLATIONS = [
    {
        "id":        "A1",
        "name":      "BCE only loss",
        "ckpt_dir":  "results/ablation/A1_bce_only",
    },
    {
        "id":        "A2",
        "name":      "Dice only loss",
        "ckpt_dir":  "results/ablation/A2_dice_only",
    },
    {
        "id":        "A3",
        "name":      "No augmentation",
        "ckpt_dir":  "results/ablation/A3_no_aug",
    },
    {
        "id":        "A4",
        "name":      "High LR (1e-3)",
        "ckpt_dir":  "results/ablation/A4_high_lr",
    },
    {
        "id":        "A5",
        "name":      "+ Prompt encoder unfrozen",
        "ckpt_dir":  "results/ablation/A5_prompt_encoder",
    },
]


def find_test_pairs(data_root="kvasir-seg"):
    image_dir = Path(data_root) / "test" / "images"
    mask_dir  = Path(data_root) / "test" / "masks"
    masks_by_stem = {
        p.stem: p for p in sorted(mask_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }
    pairs = []
    for img_path in sorted(image_dir.iterdir()):
        if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
            mask_path = masks_by_stem.get(img_path.stem)
            if mask_path:
                pairs.append((img_path, mask_path))
    return pairs


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 127


def mask_to_box(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def compute_dice(pred, gt, eps=1e-7):
    p, g = pred.astype(bool), gt.astype(bool)
    return float((2.0 * np.logical_and(p, g).sum() + eps) / (p.sum() + g.sum() + eps))


def compute_iou(pred, gt, eps=1e-7):
    p, g = pred.astype(bool), gt.astype(bool)
    return float((np.logical_and(p, g).sum() + eps) / (np.logical_or(p, g).sum() + eps))


def evaluate_checkpoint(ckpt_path, sam_registry, base_ckpt, device, pairs):
    sam = sam_registry["vit_b"](checkpoint=base_ckpt)
    state_dict = torch.load(ckpt_path, map_location=device)
    sam.load_state_dict(state_dict)
    sam.to(device)
    sam.eval()

    from segment_anything import SamPredictor
    predictor = SamPredictor(sam)

    dice_scores, iou_scores = [], []
    for img_path, mask_path in pairs:
        image   = load_rgb(img_path)
        gt_mask = load_mask(mask_path)
        bbox    = mask_to_box(gt_mask)
        if bbox is None:
            continue
        predictor.set_image(image)
        masks, _, _ = predictor.predict(box=bbox, multimask_output=False)
        pred_mask = masks[0].astype(bool)
        dice_scores.append(compute_dice(pred_mask, gt_mask))
        iou_scores.append(compute_iou(pred_mask, gt_mask))

    return float(np.mean(dice_scores)), float(np.mean(iou_scores))


def print_table(results):
    col_w = 45
    header = f"{'Experiment':<{col_w}} {'Mean Dice':>10} {'Mean IoU':>10} {'ΔDice':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    base_dice = BASELINE["mean_dice"]
    print(f"{'Baseline (BCE+Dice, lr=1e-4, aug, dec only)':<{col_w}} "
          f"{base_dice:>10.4f} {BASELINE['mean_iou']:>10.4f} {'—':>8}")
    print("-" * len(header))

    for r in results:
        delta = r["mean_dice"] - base_dice
        sign  = "+" if delta >= 0 else ""
        print(f"{r['id'] + ' ' + r['name']:<{col_w}} "
              f"{r['mean_dice']:>10.4f} {r['mean_iou']:>10.4f} {sign+f'{delta:.4f}':>8}")

    print("=" * len(header))


def main():
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    base_ckpt  = "checkpoints/sam_vit_b_01ec64.pth"
    data_root  = "kvasir-seg"

    try:
        from segment_anything import sam_model_registry
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Run: pip install -r requirements.txt")

    pairs = find_test_pairs(data_root)
    print(f"Test pairs found: {len(pairs)}")

    results = []
    for exp in ABLATIONS:
        ckpt_path = os.path.join(exp["ckpt_dir"], "sam_finetuned_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {exp['id']} — checkpoint not found: {ckpt_path}")
            continue

        print(f"Evaluating {exp['id']}: {exp['name']} ...")
        mean_dice, mean_iou = evaluate_checkpoint(
            ckpt_path, sam_model_registry, base_ckpt, device, pairs
        )
        results.append({**exp, "mean_dice": mean_dice, "mean_iou": mean_iou})
        print(f"  Mean Dice: {mean_dice:.4f}  Mean IoU: {mean_iou:.4f}")

    print_table(results)

    out_path = "results/ablation/ablation_summary.txt"
    os.makedirs("results/ablation", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Ablation Study Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Experiment':<45} {'Mean Dice':>10} {'Mean IoU':>10} {'Delta Dice':>12}\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Baseline':<45} {BASELINE['mean_dice']:>10.4f} {BASELINE['mean_iou']:>10.4f} {'—':>12}\n")
        for r in results:
            delta = r["mean_dice"] - BASELINE["mean_dice"]
            sign  = "+" if delta >= 0 else ""
            f.write(f"{r['id']+' '+r['name']:<45} {r['mean_dice']:>10.4f} "
                    f"{r['mean_iou']:>10.4f} {sign+f'{delta:.4f}':>12}\n")
    print(f"\nSummary saved to: {out_path}")


if __name__ == "__main__":
    main()
