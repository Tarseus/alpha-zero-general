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

# python scripts/plot_action_change_ratio.py \
#   --data-dir ./robust_vs_baseline_data \
#   --sims 25 50 100 200 \
#   --out-path ./compare/robust_vs_baseline_change_ratio.png

# python scripts/plot_entropy_bucket_stats.py \
#   --data-dir ./robust_vs_baseline_data \
#   --sims 25 50 100 200 \
#   --bins 0.0 0.33 0.66 1.01 \
#   --out-dir ./robust_vs_baseline_plots

# python scripts/plot_delta_q_vs_sims.py \
#   --data-dir ./robust_vs_baseline_data \
#   --sims 25 50 100 200 \
#   --out-path ./compare/robust_vs_baseline_deltaQ_vs_sims.png

# python scripts/plot_changed_vs_nochange_games.py \
#   --data-dir ./robust_vs_baseline_data \
#   --sims 25 50 100 200 \
#   --out-path ./compare/robust_vs_baseline_changed_vs_nochange_games.png

python scripts/plot_qgap_bucket_stats.py \
  --data-dir ./robust_vs_baseline_data \
  --sims 25 50 100 200 \
  --gap-bins 0.0 0.01 0.05 1.0 \
  --out-dir ./robust_vs_baseline_plots
# --gap-bins 0.0 0.02 0.05 0.1 0.2 0.4 0.7 1.0 \
python scripts/plot_phase_change_ratio.py \
  --data-dir ./robust_vs_baseline_data \
  --sims 25 50 100 200 \
  --phase-bins 0.0 0.3 0.6 1.01 \
  --out-path ./robust_vs_baseline_plots/phase_change_ratio_all_sims.png
