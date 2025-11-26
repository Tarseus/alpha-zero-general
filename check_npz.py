import numpy as np

data = np.load('robust_vs_baseline_data/robust_vs_baseline_sims25.npz')

print(sum(data['outcome']), len(data['outcome']))