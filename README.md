# YZV416E Computer Vision - Project: SAM-Med

**Adapting Segment Anything - Decoder Fine-tuning on Domain-Specific Datasets (Medical Image Segmentation)**

**Team:** SAM-Med
- **Omer Faruk Satik**: Core Codebase & Data Pipeline Lead
- **Bedirhan Ozturk**: Evaluation & Inference Lead
- **Abdullah Aydogan**: Model Training & Optimization Lead

---

## Project Summary

This project adapts Meta's **Segment Anything Model (SAM)** architecture (ViT-B) to the medical domain by fine-tuning it on the **Kvasir-SEG** dataset for pixel-level segmentation of gastrointestinal polyps in endoscopic images. To respect hardware constraints (e.g. 16 GB VRAM), the heavy ViT image encoder is fully frozen and only the lightweight mask decoder is trained.

---

## Setup and Getting Started

Follow the steps below in order to run the project from scratch on your local machine or on Colab.

### 0. Clone the Repository

```bash
git clone https://github.com/aydogn/Computer-Vision-Project.git
cd Computer-Vision-Project
```

### 1. Create and Activate the Conda Environment

```bash
conda create -n sam-med python=3.10 -y
conda activate sam-med
```

### 2. Install Dependencies

If using a GPU, install CUDA-enabled PyTorch first:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Then install the remaining project dependencies:

```bash
pip install -r requirements.txt
```

Meta's **segment-anything** package is installed automatically via the `git+https://...` entry in `requirements.txt`. NumPy and OpenCV versions are pinned for compatibility with the current PyTorch environment.

### 3. Download Dataset and Model Weights

Download the Kvasir-SEG dataset from Hugging Face:

```bash
python download_data.py
```

Download the SAM ViT-B pretrained checkpoint — Windows PowerShell:

```powershell
mkdir checkpoints
Invoke-WebRequest -Uri https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -OutFile checkpoints\sam_vit_b_01ec64.pth
```

Linux or Colab:

```bash
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P checkpoints/
```

### 4. Verify the Installation

Test that the data pipeline works correctly:

```bash
python test_dataset.py
```

If successful, a sample output image `test_output_v2.png` will be created in the project root.

### 5. Run the Zero-Shot Baseline

Evaluate the pretrained SAM ViT-B on the Kvasir-SEG test split without any fine-tuning:

```bash
python evaluate_zero_shot.py
```

Quick smoke test:

```bash
python evaluate_zero_shot.py --max-images 5 --num-visualizations 5
```

Default paths:

```text
Dataset root: kvasir-seg
Checkpoint:   checkpoints/sam_vit_b_01ec64.pth
Output dir:   results
```

Outputs:

```text
results/zero_shot_baseline.csv
results/zero_shot_summary.txt
results/zero_shot_visualizations/
```

### 6. Fine-tune the Decoder

Train only the SAM mask decoder on Kvasir-SEG (encoder frozen):

```bash
python train.py
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 20 | Number of training epochs |
| `--batch-size` | 2 | Batch size |
| `--lr` | 1e-4 | Learning rate (AdamW) |
| `--checkpoint` | `checkpoints/sam_vit_b_01ec64.pth` | SAM base weights |
| `--output-dir` | `results/finetune` | Directory for checkpoints and TensorBoard logs |

Monitor training with TensorBoard:

```bash
tensorboard --logdir results/finetune/tensorboard
```

### 7. Evaluate the Fine-tuned Model

Run inference on the test split using the fine-tuned decoder:

```bash
python evaluate_finetuned.py
```

Outputs saved to `results/finetuned_eval/`:

```text
finetuned_summary.txt          — mean/median Dice & IoU + comparison with zero-shot baseline
finetuned_results.csv          — per-image Dice and IoU scores
visualizations/                — qualitative examples (Image | Ground Truth | Fine-Tuned SAM | Overlay)
worst_cases_error_analysis.png — 5 lowest-Dice samples for error analysis
```

---

## Results

### Zero-Shot Baseline

Evaluated on the Kvasir-SEG test split (100 images) using ground-truth bounding box prompts:

```text
Evaluated images: 100
Mean Dice:        0.818568
Median Dice:      0.931399
Mean IoU:         0.759482
Median IoU:       0.871606
```

### Fine-Tuned SAM (Decoder Only)

Training configuration: 20 epochs, AdamW lr=1e-4, BCE+Dice loss, batch size=2, encoder frozen.

```text
Evaluated images: 100
Mean Dice:        0.937444
Median Dice:      0.960989
Mean IoU:         0.892464
Median IoU:       0.924908
```

### Comparison

| Model | Mean Dice | Mean IoU |
|---|---|---|
| Zero-Shot SAM ViT-B | 0.8186 | 0.7595 |
| Fine-Tuned SAM ViT-B (decoder only) | **0.9374** | **0.8925** |
| Improvement | **+11.9 pts** | **+13.3 pts** |

Only 4.3% of total parameters (mask decoder) were trained.

---

## Zero-Shot Baseline Approach

The pretrained SAM ViT-B model was evaluated without any weight updates to establish a reference for the fine-tuned model:

- Pretrained SAM ViT-B checkpoint loaded.
- 100 images from the Kvasir-SEG `test` split used.
- Bounding box prompt extracted from each ground-truth mask.
- Single mask prediction obtained via `SamPredictor` with `multimask_output=False`.
- Dice and IoU computed against ground-truth masks.
- Qualitative visualizations saved for the first 10 samples.

---

## Project Roadmap

### Completed

* **Phase 1: Environment Setup & Data Pipeline** (Omer Faruk Satik)
  - GitHub repository and Conda environment configured.
  - Kvasir-SEG download script (`download_data.py`) written.
  - SAM ViT-B checkpoint integrated.
  - PyTorch `Dataset` class written to read and match images with masks.
  - Masks forced to binary (0/1) tensor format.

* **Phase 2: Data Preprocessing** (Omer Faruk Satik)
  - Migrated from `torchvision` to `albumentations` for synchronized transforms.
  - `LongestMaxSize(1024)` and `PadIfNeeded` added to preserve polyp aspect ratio.
  - Online augmentation for the training set: `HorizontalFlip`, `VerticalFlip`, `RandomRotate90`, `ColorJitter`.
  - ImageNet normalization applied using SAM's standard mean and std values.
  - PyTorch `DataLoader` architecture built with `batch_size=2` to prevent OOM errors.

* **Phase 2 Extension: Zero-Shot Baseline** (Bedirhan Ozturk)
  - `evaluate_zero_shot.py` script added.
  - Zero-shot inference run on 100 test images.
  - Bounding box prompts extracted from ground-truth masks.
  - Dice and IoU computed and saved to `results/`.
  - Baseline result: Mean Dice `0.818568`, Mean IoU `0.759482`.

* **Phase 3: Fine-Tuning** (Abdullah Aydogan)
  - SAM image encoder (ViT) fully frozen (`requires_grad = False`).
  - Only mask decoder trained (4.3% of total parameters).
  - BCE + Dice loss defined for medical segmentation.
  - AdamW optimizer with cosine annealing scheduler.
  - TensorBoard logging for loss, Dice, IoU, and learning rate.
  - Best checkpoint saved based on validation Dice.
  - Final result: Val Dice `0.9284` after 20 epochs.

* **Phase 4: Evaluation & Visualization** (Bedirhan Ozturk)
  - `evaluate_finetuned.py` script added.
  - Fine-tuned model evaluated on the same 100 test images.
  - Dice and IoU compared against zero-shot baseline: +11.9 / +13.3 points.
  - Qualitative visualizations (Image | Ground Truth | Fine-Tuned SAM | Overlay) generated.
  - Error analysis: 5 worst-case samples identified and visualized.

### Upcoming

* **Phase 5: Finalization, Code Cleanup & Report** (All Members)
  - Modular `.py` files cleaned up or consolidated into `Final_Notebook.ipynb`.
  - README updated with final metrics and before/after visuals.
  - Final report and presentation slides prepared.
  - Hardware-friendly engineering decisions and SAM decoder fine-tuning strategy highlighted.

---

## File and Output Summary

```text
train.py                                  — fine-tuning training loop
evaluate_zero_shot.py                     — zero-shot baseline evaluation
evaluate_finetuned.py                     — fine-tuned model evaluation
dataset.py                                — PyTorch Dataset and DataLoader
download_data.py                          — Kvasir-SEG download from Hugging Face
prepare_dataset.py                        — train/val/test split creation
results/zero_shot_baseline.csv            — per-image zero-shot results
results/zero_shot_summary.txt             — zero-shot summary metrics
results/finetuned_eval/finetuned_results.csv     — per-image fine-tuned results
results/finetuned_eval/finetuned_summary.txt     — fine-tuned summary + comparison
results/finetuned_eval/visualizations/           — qualitative output images
results/finetuned_eval/worst_cases_error_analysis.png — error analysis visual
```
