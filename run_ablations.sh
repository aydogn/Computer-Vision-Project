#!/bin/bash
# Runs all 5 ablation experiments sequentially.
# Activate your conda environment before running:
#   conda activate cv-project
#   bash run_ablations.sh

set -e

EPOCHS=20
BASE_CKPT="checkpoints/sam_vit_b_01ec64.pth"
DATA="kvasir-seg"

echo "========================================"
echo "A1: BCE only loss"
echo "========================================"
python train_ablation.py \
  --experiment-name "A1_bce_only" \
  --bce-weight 1.0 --dice-weight 0.0 \
  --epochs $EPOCHS \
  --checkpoint $BASE_CKPT \
  --data-root $DATA \
  --output-dir results/ablation/A1_bce_only

echo "========================================"
echo "A2: Dice only loss"
echo "========================================"
python train_ablation.py \
  --experiment-name "A2_dice_only" \
  --bce-weight 0.0 --dice-weight 1.0 \
  --epochs $EPOCHS \
  --checkpoint $BASE_CKPT \
  --data-root $DATA \
  --output-dir results/ablation/A2_dice_only

echo "========================================"
echo "A3: No augmentation"
echo "========================================"
python train_ablation.py \
  --experiment-name "A3_no_aug" \
  --no-aug \
  --epochs $EPOCHS \
  --checkpoint $BASE_CKPT \
  --data-root $DATA \
  --output-dir results/ablation/A3_no_aug

echo "========================================"
echo "A4: High learning rate (1e-3)"
echo "========================================"
python train_ablation.py \
  --experiment-name "A4_high_lr" \
  --lr 1e-3 \
  --epochs $EPOCHS \
  --checkpoint $BASE_CKPT \
  --data-root $DATA \
  --output-dir results/ablation/A4_high_lr

echo "========================================"
echo "A5: Unfreeze prompt encoder"
echo "========================================"
python train_ablation.py \
  --experiment-name "A5_prompt_encoder" \
  --unfreeze-prompt-encoder \
  --epochs $EPOCHS \
  --checkpoint $BASE_CKPT \
  --data-root $DATA \
  --output-dir results/ablation/A5_prompt_encoder

echo ""
echo "========================================"
echo "All experiments done. Generating summary..."
echo "========================================"
python ablation_summary.py
