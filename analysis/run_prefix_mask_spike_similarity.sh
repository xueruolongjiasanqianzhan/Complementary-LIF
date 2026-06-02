#!/usr/bin/env bash
# Prefix-masked spike-train similarity analysis command.
#
# Fill in the six required paths below before running this file:
#   bash analysis/run_prefix_mask_spike_similarity.sh
#
# Notes:
# - Use checkpoint_max.pth from the LIF and LSLIF log directories.
# - Use the args.txt that matches each checkpoint's experiment settings. If the
#   run was resumed, the args.txt from the resumed run is fine as long as the
#   model/neuron settings are the same as the analyzed checkpoint.

set -euo pipefail

LIF_CHECKPOINT="/path/to/lif_log/checkpoint_max.pth"
LSLIF_CHECKPOINT="/path/to/lslif_log/checkpoint_max.pth"
LIF_ARGS="/path/to/lif_log/args.txt"
LSLIF_ARGS="/path/to/lslif_log/args.txt"
DATA_DIR="/path/to/DVS-CIFAR10"
OUT_DIR="analysis_results/prefix_mask_dvscifar10_lif_vs_lslif"

python analysis/prefix_mask_spike_similarity.py \
  --lif-checkpoint "${LIF_CHECKPOINT}" \
  --lsl-checkpoint "${LSLIF_CHECKPOINT}" \
  --lif-args "${LIF_ARGS}" \
  --lsl-args "${LSLIF_ARGS}" \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}" \
  --T 16 \
  --mask-prefixes 2 4 8 \
  --batch-size 16 \
  --workers 0 \
  --layer-mode shallow_middle_deep
