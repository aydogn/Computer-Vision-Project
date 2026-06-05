import argparse
import csv
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned SAM ViT-B decoder with Morphological Post-Processing."
    )
    parser.add_argument("--data-root", default="kvasir-seg", help="Kvasir-SEG root directory.")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join("checkpoints", "sam_vit_b_01ec64.pth"),
        help="Path to the original SAM ViT-B checkpoint.",
    )
    parser.add_argument(
        "--finetuned-checkpoint",
        default=os.path.join("models", "sam_finetuned_best.pth"),
        help="Path to the fine-tuned decoder weights (.pth).",
    )
    parser.add_argument("--model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--output-dir", default=os.path.join("results", "finetuned_eval_postprocess"), help="Directory for Post-Process evaluation outputs.")
    parser.add_argument(
        "--num-visualizations",
        type=int,
        default=10,
        help="Number of qualitative examples to save.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit for quick smoke tests.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device used for SAM inference.",
    )
    return parser.parse_args()


def find_image_mask_pairs(data_root):
    image_dir = Path(data_root) / "test" / "images"
    mask_dir = Path(data_root) / "test" / "masks"

    if not image_dir.exists():
        raise FileNotFoundError(f"Test image directory not found: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Test mask directory not found: {mask_dir}")

    masks_by_stem = {
        path.stem: path
        for path in sorted(mask_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }

    pairs = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        mask_path = masks_by_stem.get(image_path.stem)
        if mask_path is not None:
            pairs.append((image_path, mask_path))

    return pairs


def load_rgb_image(image_path):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_binary_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return mask > 127


def mask_to_box(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def compute_dice(pred_mask, gt_mask, eps=1e-7):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    denominator = pred.sum() + gt.sum()
    return float((2.0 * intersection + eps) / (denominator + eps))


def compute_iou(pred_mask, gt_mask, eps=1e-7):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((intersection + eps) / (union + eps))


def apply_postprocess(mask_bool):
    """
    Applies OpenCV morphological operations to clean up the mask.
    (Removes noise, fills holes, keeps largest blob, smooths edges)
    """
    mask_uint8 = (mask_bool * 255).astype(np.uint8)
    
    # 1. Opening: Remove small white noise (islands)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_open)
    
    # 2. Hole filling: fill all dark holes inside the predicted polyp region.
    # We use external contours and fill their interiors completely.
    contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = mask_opened.copy()
    cv2.drawContours(mask_filled, contours, -1, 255, thickness=cv2.FILLED)
    
    # 3. Do not keep only the largest component, because some images contain multiple polyps.
    mask_final = mask_filled
    
    # 4. Smoothing boundaries: Gaussian blur and threshold
    mask_blurred = cv2.GaussianBlur(mask_final, (7, 7), 0)
    _, mask_smoothed = cv2.threshold(mask_blurred, 127, 255, cv2.THRESH_BINARY)
        
    return mask_smoothed > 127


def save_visualization(image, gt_mask, pred_mask, pred_mask_pp, output_path, title):
    overlay_sam = image.copy()
    overlay_sam[gt_mask] = (0.55 * overlay_sam[gt_mask] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
    overlay_sam[pred_mask] = (0.55 * overlay_sam[pred_mask] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)

    overlay_pp = image.copy()
    overlay_pp[gt_mask] = (0.55 * overlay_pp[gt_mask] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
    overlay_pp[pred_mask_pp] = (0.55 * overlay_pp[pred_mask_pp] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)

    plt.figure(figsize=(18, 4))

    plt.subplot(1, 5, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(gt_mask, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Raw SAM Mask")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(pred_mask_pp, cmap="gray")
    plt.title("Post-Processed Mask")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(overlay_pp)
    plt.title("Overlay (Post-Process)")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_worst_cases_visualization(worst_cases, output_path):
    n = len(worst_cases)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = [axes]

    for i, (image, gt_mask, raw_mask, pp_mask, dice, dice_pp, filename) in enumerate(worst_cases):
        axes[i][0].imshow(image)
        axes[i][0].set_title(f"{filename}")
        axes[i][0].axis("off")

        axes[i][1].imshow(gt_mask, cmap="gray")
        axes[i][1].set_title("Ground Truth")
        axes[i][1].axis("off")

        axes[i][2].imshow(raw_mask, cmap="gray")
        axes[i][2].set_title(f"Raw SAM (Dice: {dice:.4f})")
        axes[i][2].axis("off")

        axes[i][3].imshow(pp_mask, cmap="gray")
        axes[i][3].set_title(f"Post-Processed (Dice: {dice_pp:.4f})")
        axes[i][3].axis("off")

    plt.suptitle("Lowest Dice Score Samples (Error Analysis with Post-Processing)", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_csv(rows, output_path):
    fieldnames = [
        "filename", "dice_raw", "iou_raw", "dice_pp", "iou_pp", "sam_score"
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows, skipped_count, args, output_path):
    dice_raw = np.array([r["dice_raw"] for r in rows], dtype=np.float32)
    iou_raw  = np.array([r["iou_raw"] for r in rows], dtype=np.float32)
    dice_pp  = np.array([r["dice_pp"] for r in rows], dtype=np.float32)
    iou_pp   = np.array([r["iou_pp"] for r in rows], dtype=np.float32)

    lines = [
        "Fine-Tuned SAM Evaluation (with Morphological Post-Processing)",
        f"Model: {args.model_type}",
        f"Device: {args.device}",
        f"Evaluated images: {len(rows)}",
        f"Skipped images: {skipped_count}",
        "",
        "--- RAW SAM PREDICTIONS ---",
        f"Mean Dice:   {dice_raw.mean():.6f}",
        f"Median Dice: {np.median(dice_raw):.6f}",
        f"Mean IoU:    {iou_raw.mean():.6f}",
        f"Median IoU:  {np.median(iou_raw):.6f}",
        "",
        "--- POST-PROCESSING (OpenCV Morphology) ---",
        f"Mean Dice:   {dice_pp.mean():.6f} (Change: {dice_pp.mean() - dice_raw.mean():+.6f})",
        f"Median Dice: {np.median(dice_pp):.6f}",
        f"Mean IoU:    {iou_pp.mean():.6f} (Change: {iou_pp.mean() - iou_raw.mean():+.6f})",
        f"Median IoU:  {np.median(iou_pp):.6f}",
    ]

    with open(output_path, "w", encoding="utf-8") as summary_file:
        summary_file.write("\n".join(lines) + "\n")


def main():
    args = parse_args()

    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ModuleNotFoundError:
        raise ModuleNotFoundError("pip install -r requirements.txt")

    if not os.path.exists(args.finetuned_checkpoint):
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {args.finetuned_checkpoint}")

    pairs = find_image_mask_pairs(args.data_root)
    if args.max_images is not None:
        pairs = pairs[: args.max_images]

    output_dir = Path(args.output_dir)
    visualization_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SAM base model and fine-tuned weights...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    state_dict = torch.load(args.finetuned_checkpoint, map_location=args.device)
    sam.load_state_dict(state_dict)
    sam.to(device=args.device)
    sam.eval()
    predictor = SamPredictor(sam)

    rows = []
    skipped_count = 0
    worst_cases_data = []

    print(f"Evaluating {len(pairs)} test images on {args.device} with Post-Processing...")
    for index, (image_path, mask_path) in enumerate(pairs, start=1):
        image = load_rgb_image(image_path)
        gt_mask = load_binary_mask(mask_path)
        bbox = mask_to_box(gt_mask)

        if bbox is None:
            skipped_count += 1
            continue

        # 1. SAM prediction
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(box=bbox, multimask_output=False)
        pred_mask_raw = masks[0].astype(bool)

        # 2. OpenCV Post-Processing
        pred_mask_pp = apply_postprocess(pred_mask_raw)

        # 3. Metric computation
        dice_raw = compute_dice(pred_mask_raw, gt_mask)
        iou_raw  = compute_iou(pred_mask_raw, gt_mask)
        
        dice_pp = compute_dice(pred_mask_pp, gt_mask)
        iou_pp  = compute_iou(pred_mask_pp, gt_mask)

        rows.append({
            "filename": image_path.name,
            "dice_raw": dice_raw,
            "iou_raw": iou_raw,
            "dice_pp": dice_pp,
            "iou_pp": iou_pp,
            "sam_score": float(scores[0])
        })

        worst_cases_data.append((image, gt_mask, pred_mask_raw, pred_mask_pp, dice_raw, dice_pp, image_path.name))

        # Save visualizations for the first N images.
        if len(rows) <= args.num_visualizations:
            vis_path = visualization_dir / f"{len(rows):03d}_{image_path.stem}.png"
            save_visualization(
                image, gt_mask, pred_mask_raw, pred_mask_pp, vis_path,
                title=f"{image_path.name} | Raw Dice: {dice_raw:.4f} | PP Dice: {dice_pp:.4f}"
            )

        if index % 10 == 0 or index == len(pairs):
            print(f"Processed: {index}/{len(pairs)}")

    # Select the 5 lowest-scoring images by raw Dice.
    worst_cases_data.sort(key=lambda x: x[4]) 
    worst_5 = worst_cases_data[:5]
    worst_path = output_dir / "worst_cases_error_analysis_pp.png"
    save_worst_cases_visualization(worst_5, worst_path)

    # Save results.
    write_csv(rows, output_dir / "postprocess_results.csv")
    write_summary(rows, skipped_count, args, output_dir / "postprocess_summary.txt")

    print("\nPost-processing evaluation completed.")
    print(f"Visualizations and scores were saved to: {output_dir}")

if __name__ == "__main__":
    main()
