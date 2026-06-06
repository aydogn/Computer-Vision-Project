# YZV416E Computer Vision - Project: SAM-Med

**Adapting Segment Anything - Decoder Fine-tuning on Domain-Specific Datasets (Medical Image Segmentation)**

**Team:** SAM-Med

- **Ömer Faruk Satık:** Core Codebase & Data Pipeline Lead
- **Bedirhan Öztürk:** Evaluation & Inference Lead
- **Abdullah Aydoğan:** Model Training & Optimization Lead

---

## Project Summary

This project adapts Meta's **Segment Anything Model (SAM) ViT-B** to gastrointestinal polyp segmentation on **Kvasir-SEG**. To keep the solution hardware-friendly, the SAM image encoder is frozen and only the lightweight mask decoder is fine-tuned. The project includes zero-shot evaluation, decoder fine-tuning, ablation studies, prompt-dependence analysis, and post-processing experiments.

The final recommended model is the **20-epoch decoder-only fine-tuned SAM ViT-B** checkpoint. It trains only **4.3%** of SAM's parameters and improves mean Dice from **0.8186** to **0.9374** on the held-out test split.

---

## Setup

### 1. Create Environment

```bash
conda create -n sam-med python=3.10 -y
conda activate sam-med
```

### 2. Install PyTorch

Install the CUDA-enabled PyTorch version suitable for your system. Example for CUDA 12.6:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 3. Install Project Dependencies

```bash
pip install -r requirements.txt
```

For the optional Dense CRF post-processing experiment, install `pydensecrf` separately if needed:

```bash
conda install -c conda-forge pydensecrf
```

`python-docx` is not required for running the project code.

### 4. Download Dataset

```bash
python download_data.py
```

Expected split after preparation/download:

| Split | Images | Masks |
|---|---:|---:|
| Train | 800 | 800 |
| Validation | 100 | 100 |
| Test | 100 | 100 |

### 5. Download SAM ViT-B Checkpoint

Windows PowerShell:

```powershell
mkdir checkpoints
Invoke-WebRequest -Uri https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -OutFile checkpoints\sam_vit_b_01ec64.pth
```

Linux / Colab:

```bash
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P checkpoints/
```

### 6. Download Fine-Tuned Model Checkpoint (Optional)

If you want to skip the training phase and directly evaluate our best model, you can download the fine-tuned weights here:

- **[Download sam_finetuned_best.pth from Google Drive](https://drive.google.com/file/d/1KJAkPOuRXVvDiaVD-G_4SfcXii7NjvTm/view?usp=sharing)**

Please place the downloaded file at the following path before running evaluation scripts:
`results/finetune/sam_finetuned_best.pth`

---

## Main Commands

### Verify Dataset Pipeline

```bash
python test_dataset.py
```

### Zero-Shot Baseline

```bash
python evaluate_zero_shot.py
```

Quick smoke test:

```bash
python evaluate_zero_shot.py --max-images 5 --num-visualizations 5
```

### Fine-Tune SAM Decoder

```bash
python train.py
```

Default training configuration:

| Setting | Value |
|---|---|
| Base model | SAM ViT-B |
| Frozen modules | Image encoder, prompt encoder |
| Trainable module | Mask decoder only |
| Trainable parameters | 4,058,340 / 93,735,472 (4.3%) |
| Loss | 0.5 BCE + 0.5 Dice |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Epochs | 20 |
| Batch size | 2 |

### Evaluate Fine-Tuned Model

```bash
python evaluate_finetuned.py
```

### Ablation Studies

Run the ablation experiments:

```bash
bash run_ablations.sh
```

Summarize ablation results:

```bash
python ablation_summary.py
```

### Promptless Training Experiment

```bash
python train_promptless.py
```

### Post-Processing Experiments

Morphological post-processing:

```bash
python evaluate_postprocess.py --finetuned-checkpoint results/finetune/sam_finetuned_best.pth
```

Dense CRF post-processing:

```bash
python evaluate_finetuned_crf.py --finetuned-checkpoint results/finetune/sam_finetuned_best.pth
```

---

## Evaluation Protocol

All main comparisons use the same Kvasir-SEG test split of 100 images. SAM is evaluated with **ground-truth bounding box prompts** extracted from the target masks. This is a prompt-conditioned segmentation setup, not a fully automatic detector. The same prompt protocol is used for zero-shot and fine-tuned evaluations to keep comparisons fair.

Metrics:

- **Dice coefficient** for mask overlap.
- **IoU** for stricter intersection-over-union evaluation.

---

## Final Results

### Main Comparison

| Model | Mean Dice | Median Dice | Mean IoU | Median IoU |
|---|---:|---:|---:|---:|
| Zero-shot SAM ViT-B | 0.8186 | 0.9314 | 0.7595 | 0.8716 |
| Fine-tuned SAM decoder, 20 epochs | **0.9374** | **0.9610** | **0.8925** | **0.9249** |
| Improvement | **+0.1189** | - | **+0.1330** | - |

### Full Experiment Summary

| Rank | Model | Mean Dice | Mean IoU | Delta Dice vs Zero-Shot |
|---:|---|---:|---:|---:|
| 1 | Fine-tuned + Morphological post-processing | 0.9384 | 0.8938 | +0.1198 |
| 2 | Fine-tuned SAM decoder, 20 epochs (primary) | 0.9374 | 0.8925 | +0.1189 |
| 3 | Extended fine-tuning (+15 epochs) | 0.9373 | 0.8934 | +0.1188 |
| 4 | Prompt encoder unfrozen | 0.9364 | 0.8868 | +0.1178 |
| 5 | Dice-only loss | 0.9327 | 0.8823 | +0.1141 |
| 6 | No augmentation | 0.9319 | 0.8810 | +0.1133 |
| 7 | BCE-only loss | 0.9271 | 0.8736 | +0.1085 |
| 8 | High learning rate (1e-3) | 0.9184 | 0.8601 | +0.0998 |
| 9 | Fine-tuned + Dense CRF | 0.8670 | 0.7860 | +0.0484 |
| - | Zero-shot SAM ViT-B | 0.8186 | 0.7595 | - |
| 10 | Promptless fine-tuning | 0.7772 | 0.6795 | -0.0413 |

Although morphology has the numerically highest Dice, the gain is only **+0.0006** over raw fine-tuned SAM on the post-processing subset. The final recommended model remains the **20-epoch decoder-only fine-tuned SAM** because it is simpler and its performance is effectively identical.

---

## Key Findings

- Decoder-only fine-tuning substantially improves SAM on medical polyp segmentation.
- Training only the mask decoder is efficient: only 4.3% of model parameters are updated.
- BCE + Dice loss outperforms BCE-only and Dice-only variants.
- Data augmentation is important for generalization; removing it causes overfitting.
- A high learning rate (`1e-3`) damages adaptation performance.
- Unfreezing the prompt encoder provides no meaningful gain.
- Promptless training performs worse than zero-shot, confirming that SAM strongly depends on prompts.
- Morphological post-processing gives a negligible gain.
- Dense CRF degrades performance because endoscopic images have weak colour contrast and specular artifacts.

---

## Important Files

### Code

```text
dataset.py                         PyTorch Dataset and DataLoader utilities
download_data.py                   Kvasir-SEG download script
prepare_dataset.py                 Reproducible 800/100/100 split helper
test_dataset.py                    Dataset pipeline smoke test
evaluate_zero_shot.py              Zero-shot SAM baseline evaluation
train.py                           Primary decoder fine-tuning script
evaluate_finetuned.py              Primary fine-tuned model evaluation
train_ablation.py                  Controlled ablation training script
run_ablations.sh                   Ablation runner
ablation_summary.py                Ablation result summarizer
train_promptless.py                Empty-prompt training experiment
evaluate_postprocess.py            Morphological post-processing evaluation
evaluate_finetuned_crf.py          Dense CRF post-processing evaluation
```

### Results and Reports

```text
results/ALL_RESULTS_SUMMARY.txt                 Condensed final result summary
results/ALL_MODELS_RESULTS.txt                  Detailed all-model comparison
results/zero_shot_summary.txt                   Zero-shot metrics
results/finetuned_eval/finetuned_summary.txt    Primary model metrics
results/ablation/ablation_summary.txt           Ablation table
YZV416E_FinalReport.docx                        Final project report
report_assets/                                  Figures used in the final report
```

Large local files such as `kvasir-seg/`, `checkpoints/`, and `.pth` model weights are intentionally excluded from version control.

---

## Project Status

Completed:

- Phase 1: environment setup and data pipeline
- Phase 2: preprocessing
- Phase 2 extension: zero-shot baseline
- Phase 3: decoder fine-tuning
- Phase 4: fine-tuned evaluation and visualization
- Phase 5: final experiments, ablations, post-processing analysis, and final report

The repository is now in final submission format.
