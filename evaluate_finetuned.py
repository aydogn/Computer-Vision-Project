import argparse
import csv
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned SAM ViT-B decoder on the Kvasir-SEG test split."
    )
    parser.add_argument("--data-root", default="kvasir-seg", help="Kvasir-SEG root directory.")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join("checkpoints", "sam_vit_b_01ec64.pth"),
        help="Path to the original SAM ViT-B checkpoint.",
    )
    parser.add_argument(
        "--finetuned-checkpoint",
        default=os.path.join("results", "finetune", "sam_finetuned_best.pth"),
        help="Path to the fine-tuned decoder weights (.pth).",
    )
    parser.add_argument("--model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--output-dir", default=os.path.join("results", "finetuned_eval"), help="Directory for evaluation outputs.")
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

    if not pairs:
        raise FileNotFoundError(f"No image-mask pairs found under: {data_root}")

    return pairs


def load_rgb_image(image_path):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_binary_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask: {mask_path}")
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


def save_visualization(image, gt_mask, pred_mask, output_path, title):
    overlay = image.copy()
    overlay[gt_mask] = (0.55 * overlay[gt_mask] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
    overlay[pred_mask] = (0.55 * overlay[pred_mask] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(gt_mask, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Fine-Tuned SAM")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_worst_cases_visualization(worst_cases, output_path):
    """Plot the 5 lowest-Dice samples side by side for error analysis."""
    n = len(worst_cases)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    for i, (image, gt_mask, pred_mask, dice, filename) in enumerate(worst_cases):
        axes[i][0].imshow(image)
        axes[i][0].set_title(f"{filename}\nDice: {dice:.4f}")
        axes[i][0].axis("off")

        axes[i][1].imshow(gt_mask, cmap="gray")
        axes[i][1].set_title("Ground Truth")
        axes[i][1].axis("off")

        axes[i][2].imshow(pred_mask, cmap="gray")
        axes[i][2].set_title("Fine-Tuned SAM")
        axes[i][2].axis("off")

    plt.suptitle("Lowest Dice Score Samples (Error Analysis)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_csv(rows, output_path):
    fieldnames = [
        "filename",
        "dice",
        "iou",
        "sam_score",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows, skipped_count, args, output_path):
    dice_values = np.array([row["dice"] for row in rows], dtype=np.float32)
    iou_values = np.array([row["iou"] for row in rows], dtype=np.float32)

    lines = [
        "Fine-Tuned SAM Evaluation",
        f"Model: {args.model_type}",
        "Prompt: ground-truth bounding box",
        f"Dataset root: {args.data_root}",
        f"Base checkpoint: {args.checkpoint}",
        f"Fine-tuned weights: {args.finetuned_checkpoint}",
        f"Device: {args.device}",
        f"Evaluated images: {len(rows)}",
        f"Skipped images: {skipped_count}",
        "",
        f"Mean Dice:   {dice_values.mean():.6f}",
        f"Median Dice: {np.median(dice_values):.6f}",
        f"Mean IoU:    {iou_values.mean():.6f}",
        f"Median IoU:  {np.median(iou_values):.6f}",
        "",
        "--- Zero-Shot Baseline (reference) ---",
        "Mean Dice:   0.818568",
        "Mean IoU:    0.759482",
        "",
        f"Dice improvement: +{dice_values.mean() - 0.818568:.6f}",
        f"IoU  improvement: +{iou_values.mean() - 0.759482:.6f}",
    ]

    with open(output_path, "w", encoding="utf-8") as summary_file:
        summary_file.write("\n".join(lines) + "\n")


def main():
    args = parse_args()

    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "segment-anything is not installed. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"SAM checkpoint not found: {args.checkpoint}")

    if not os.path.exists(args.finetuned_checkpoint):
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {args.finetuned_checkpoint}")

    pairs = find_image_mask_pairs(args.data_root)
    if args.max_images is not None:
        pairs = pairs[: args.max_images]

    output_dir = Path(args.output_dir)
    visualization_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SAM base model: {args.checkpoint}")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)

    print(f"Loading fine-tuned weights: {args.finetuned_checkpoint}")
    state_dict = torch.load(args.finetuned_checkpoint, map_location=args.device)
    sam.load_state_dict(state_dict)

    sam.to(device=args.device)
    sam.eval()
    predictor = SamPredictor(sam)

    rows = []
    skipped_count = 0
    worst_cases_data = []

    print(f"Evaluating {len(pairs)} test images on {args.device}...")
    for index, (image_path, mask_path) in enumerate(pairs, start=1):
        image = load_rgb_image(image_path)
        gt_mask = load_binary_mask(mask_path)
        bbox = mask_to_box(gt_mask)

        if bbox is None:
            skipped_count += 1
            print(f"Skipping empty mask: {mask_path.name}")
            continue

        predictor.set_image(image)
        masks, scores, _ = predictor.predict(box=bbox, multimask_output=False)

        pred_mask = masks[0].astype(bool)
        dice = compute_dice(pred_mask, gt_mask)
        iou = compute_iou(pred_mask, gt_mask)
        sam_score = float(scores[0])

        rows.append(
            {
                "filename": image_path.name,
                "dice": dice,
                "iou": iou,
                "sam_score": sam_score,
                "bbox_xmin": float(bbox[0]),
                "bbox_ymin": float(bbox[1]),
                "bbox_xmax": float(bbox[2]),
                "bbox_ymax": float(bbox[3]),
            }
        )

        worst_cases_data.append((image, gt_mask, pred_mask, dice, image_path.name))

        if len(rows) <= args.num_visualizations:
            visualization_path = visualization_dir / f"{len(rows):03d}_{image_path.stem}.png"
            save_visualization(
                image,
                gt_mask,
                pred_mask,
                visualization_path,
                title=f"{image_path.name} | Dice: {dice:.4f} | IoU: {iou:.4f}",
            )

        if index % 10 == 0 or index == len(pairs):
            print(f"Processed: {index}/{len(pairs)}")

    if not rows:
        raise RuntimeError("No images were evaluated. Check mask and dataset paths.")

    # Error analysis: 5 lowest-Dice samples
    worst_cases_data.sort(key=lambda x: x[3])
    worst_5 = worst_cases_data[:5]
    worst_path = output_dir / "worst_cases_error_analysis.png"
    save_worst_cases_visualization(worst_5, worst_path)
    print(f"\n5 lowest-Dice samples (error analysis):")
    for _, _, _, dice, fname in worst_5:
        print(f"  {fname}  Dice: {dice:.4f}")

    csv_path = output_dir / "finetuned_results.csv"
    summary_path = output_dir / "finetuned_summary.txt"
    write_csv(rows, csv_path)
    write_summary(rows, skipped_count, args, summary_path)

    mean_dice = np.mean([row["dice"] for row in rows])
    mean_iou = np.mean([row["iou"] for row in rows])
    median_dice = np.median([row["dice"] for row in rows])
    median_iou = np.median([row["iou"] for row in rows])

    print("\nFine-tuned model evaluation complete.")
    print(f"Evaluated images: {len(rows)}")
    print(f"Skipped images:   {skipped_count}")
    print(f"Mean Dice:   {mean_dice:.6f}  (Zero-Shot: 0.818568  |  Improvement: +{mean_dice - 0.818568:.6f})")
    print(f"Median Dice: {median_dice:.6f}")
    print(f"Mean IoU:    {mean_iou:.6f}  (Zero-Shot: 0.759482  |  Improvement: +{mean_iou - 0.759482:.6f})")
    print(f"Median IoU:  {median_iou:.6f}")
    print(f"CSV saved:              {csv_path}")
    print(f"Summary saved:          {summary_path}")
    print(f"Visualizations saved:   {visualization_dir}")
    print(f"Error analysis saved:   {worst_path}")


if __name__ == "__main__":
    main()
