#!/usr/bin/env bash
set -euo pipefail

# Minimal starter for the Llama fine-tuning path.
# Edit DATA_NAME and TASK2 for your experiment naming.

DATA_NAME="rswtyjs"
LR="5e-5"
EPOCH="3"
TASK1="Gen"
TASK2="full_comment"

python fine_tune/main.py \
  --data_name "$DATA_NAME" \
  --lr "$LR" \
  --epoch "$EPOCH" \
  --task1 "$TASK1" \
  --task2 "$TASK2"