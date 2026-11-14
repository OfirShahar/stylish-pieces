#!/usr/bin/env bash
# Train and test all 16 deep models of the paper on the CLEOPATRA benchmark (11 classes).
# Settings are exactly those of the paper's runs (see paper_results/cleopatra/*/args.json).
# Shared settings (train.py defaults): 50 epochs, AdamW, weight decay 1e-2, cosine schedule,
# label smoothing 0.1, mixup 0.2, early stopping patience 15, seed 42.
#
# Usage: bash scripts/reproduce_cleopatra.sh /path/to/CLEOPATRA [extra train.py args]
#   (CLEOPATRA must contain train/, valid/ and test/ folders of PNG fragments)
set -euo pipefail
DATA_DIR="${1:?usage: $0 DATA_DIR [extra train.py args]}"; shift
cd "$(dirname "$0")/.."

run() { python train.py --dataset cleopatra --data_dir "$DATA_DIR" "$@"; }

for arch in resnet50 efficientnet_b3 convnext_small vit_b_16 swin_t maxvit_t \
            dinov2_small clip_vit_b32 dinov3_small; do
  run --arch "$arch" --batch_size 32 --lr 1e-4 "$@"
done
run --arch dinov3_base        --batch_size 32 --lr 5e-5 "$@"
run --arch dinov3_large       --batch_size 16 --lr 5e-5 "$@"
run --arch mobilenet_v3_small --batch_size 64 --lr 1e-4 "$@"
run --arch mobilenet_v3_large --batch_size 64 --lr 1e-4 "$@"
run --arch mobilevit_s        --batch_size 32 --lr 5e-5 "$@"
run --arch efficientvit_b0    --batch_size 64 --lr 5e-5 "$@"
run --arch fastvit_t8         --batch_size 32 --lr 5e-5 "$@"
