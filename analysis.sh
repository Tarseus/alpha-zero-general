# nohup python post_analysis.py \
# --checkpoint ./models/baseline.pth.tar \
# --num-states 1000 \
# --gen-sims 25 \
# --teacher-sims 100 \
# --out-dir ./analysis_out \
# --device 1 \
# --out-dir ./baseline > analysis_base.out 2>&1 &

nohup python post_analysis.py \
--checkpoint ./models/best60.pth.tar \
--num-states 1000 \
--gen-sims 25 \
--teacher-sims 100 \
--out-dir ./ours \
--device 1 > analysis_ours.out 2>&1 &