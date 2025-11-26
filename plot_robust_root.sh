export CUDA_VISIBLE_DEVICES=1

# nohup python scripts/gen_robust_baseline_data.py \
#   --sims 50 100 200 \
#   --games-per-match 200 \
#   --model-dir ./models \
#   --model-file baseline.pth.tar \
#   --out-dir ./robust_vs_baseline_data > gen_robust_baseline_data.log 2>&1 &

# nohup python scripts/plot_action_change_ratio.py \
#   --data-dir ./robust_vs_baseline_data \
#   --sims 25 50 100 200 \
#   --out-path ./compare/robust_vs_baseline_change_ratio.png > action_change_ratio.log 2>&1 &

# nohup python scripts/plot_entropy_bucket_stats.py \
#   --data-dir ./robust_vs_baseline_data \
#   --sims 25 50 100 200 \
#   --bins 0.0 0.33 0.66 1.01 \
#   --out-dir ./robust_vs_baseline_plots > entropy_bucket_stats.log 2>&1 &

# python scripts/gen_robust_baseline_data.py \
#   --sims 25 50 100 200 \
#   --games-per-match 200 \
#   --model-dir ./models \
#   --model-file baseline.pth.tar \
#   --out-dir ./robust_vs_baseline_data

python scripts/plot_action_change_ratio.py \
  --data-dir ./robust_vs_baseline_data \
  --sims 25 50 100 200 \
  --out-path ./compare/robust_vs_baseline_change_ratio.png

python scripts/plot_entropy_bucket_stats.py \
  --data-dir ./robust_vs_baseline_data \
  --sims 25 50 100 200 \
  --bins 0.0 0.33 0.66 1.01 \
  --out-dir ./robust_vs_baseline_plots