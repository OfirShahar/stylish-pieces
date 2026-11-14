#!/usr/bin/env bash
# After training on the Pompeian Styles dataset: traditional-ML baselines,
# evaluation on the square-tile control set, and result tables.
#
# Usage: bash scripts/reproduce_analysis.sh /path/to/pompeian-styles
# Run scripts/reproduce_pompeian.sh first.
set -euo pipefail
DATA_DIR="${1:?usage: $0 DATA_DIR}"
cd "$(dirname "$0")/.."

# Random Forest / Gradient Boosting on frozen deep features and HOG + colour histograms
python train_random_forest.py --data_dir "$DATA_DIR" --feature_extractor all \
    --n_estimators 500 --pca_components 256

# Square-tile control test set (Section "Square Test-Set Variation")
python evaluate.py --dataset pompeian --data_dir "$DATA_DIR" --results_dir results/pompeian \
    --split testset_square --output_dir results/pompeian_testset_square

# Ranking tables
python collect_results.py --results_dir results/pompeian
if [ -d results/cleopatra ]; then
  python collect_results.py --results_dir results/cleopatra
fi
