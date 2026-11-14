#!/usr/bin/env bash
# Train and test all 16 deep models of the paper on the Pompeian Styles dataset.
# Settings are exactly those of the paper's runs (see paper_results/pompeian/*/args.json).
# Shared settings (train.py defaults): 50 epochs, AdamW, weight decay 1e-2, cosine schedule,
# label smoothing 0.1, mixup 0.2, early stopping patience 15, seed 42.
#
# Usage: bash scripts/reproduce_pompeian.sh /path/to/pompeian-styles [extra train.py args]
# DINOv3 weights are gated on Hugging Face: accept the license there and `export HF_TOKEN=...`.
set -euo pipefail
DATA_DIR="${1:?usage: $0 DATA_DIR [extra train.py args]}"; shift
cd "$(dirname "$0")/.."

run() { python train.py --dataset pompeian --data_dir "$DATA_DIR" "$@"; }

# CNNs
run --arch resnet50           --batch_size 32 --lr 1e-4 "$@"
run --arch efficientnet_b3    --batch_size 24 --lr 1e-4 "$@"
run --arch convnext_small     --batch_size 32 --lr 5e-5 "$@"
# Transformers / hybrid
run --arch vit_b_16           --batch_size 32 --lr 5e-5 "$@"
run --arch swin_t             --batch_size 32 --lr 5e-5 "$@"
run --arch maxvit_t           --batch_size 24 --lr 5e-5 "$@"
# Foundation models
run --arch dinov2_small       --batch_size 32 --lr 1e-4 "$@"
run --arch clip_vit_b32       --batch_size 64 --lr 1e-3 --weight_decay 1e-3 --mixup_alpha 0.0 "$@"  # frozen encoder, linear probe
run --arch dinov3_small       --batch_size 32 --lr 1e-4 "$@"
run --arch dinov3_base        --batch_size 32 --lr 5e-5 "$@"
run --arch dinov3_large       --batch_size 16 --lr 5e-5 "$@"
# Edge-device models
run --arch mobilenet_v3_small --batch_size 64 --lr 1e-4 "$@"
run --arch mobilenet_v3_large --batch_size 64 --lr 1e-4 "$@"
run --arch mobilevit_s        --batch_size 32 --lr 5e-5 "$@"
run --arch efficientvit_b0    --batch_size 64 --lr 5e-5 "$@"
run --arch fastvit_t8         --batch_size 32 --lr 5e-5 "$@"
