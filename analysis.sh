#!/usr/bin/env bash
set -euo pipefail

# Sequential runs to enable an apples-to-apples comparison.
# Adjust checkpoints and device indices as needed.

echo "[1/3] Running baseline analysis..."
python post_analysis.py \
  --checkpoint ./models/baseline.pth.tar \
  --num-states 1000 \
  --gen-sims 25 \
  --teacher-sims 100 \
  --out-dir ./baseline \
  --device 1

echo "[2/3] Running ours analysis..."
python post_analysis.py \
  --checkpoint ./models/best60.pth.tar \
  --num-states 1000 \
  --gen-sims 25 \
  --teacher-sims 100 \
  --out-dir ./ours \
  --device 1

echo "[3/3] Comparing results and plotting overlays..."
python scripts/compare_analysis.py \
  --dir-a ./baseline --label-a baseline \
  --dir-b ./ours     --label-b ours \
  --out-dir ./compare

echo "Done. See ./compare for overlaid plots and CSV summary."
