import pandas as pd
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    bin_total = np.bincount(binids, minlength=n_bins)
    nonzero = bin_total > 0
    bin_probs = np.bincount(binids, weights=y_prob, minlength=n_bins)[nonzero] / bin_total[nonzero]
    bin_acc = np.bincount(binids, weights=y_true, minlength=n_bins)[nonzero] / bin_total[nonzero]
    return np.sum(np.abs(bin_probs - bin_acc) * bin_total[nonzero]) / len(y_true)

df = pd.read_parquet("final_experiment_100_full.parquet")
df = df[df['label'].isin([0, 1])]

# Recover raw score from SAME (which was expit(4*score - 2))
# actually, mycompute_metrics used: p_same = sigmoid(4.0 * same_score - 2.0)
# So we need the original same_score.
# Re-calculate or use inverse sigmoid.
y_true = df['label'].values
# p = 1 / (1 + exp(-(4s - 2))) => log(p/(1-p)) = 4s - 2 => s = (logit(p) + 2)/4
p = df['SAME'].values
logit_p = np.log(p / (1 - p))
same_scores = (logit_p + 2.0) / 4.0

def objective(params):
    k, b = params
    p_calib = expit(k * same_scores + b)
    return compute_ece(y_true, p_calib)

res = minimize(objective, [4.0, -2.0], method='Nelder-Mead')
k_opt, b_calib = res.x
p_opt = expit(k_opt * same_scores + b_calib)
ece_opt = compute_ece(y_true, p_opt)
auc_opt = roc_auc_score(y_true, p_opt)

print(f"Optimal k: {k_opt:.4f}")
print(f"Optimal b: {b_calib:.4f}")
print(f"New ECE: {ece_opt:.4f}")
print(f"AUROC: {auc_opt:.4f}")
