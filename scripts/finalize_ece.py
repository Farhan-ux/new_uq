import pandas as pd
import numpy as np
import scipy.special

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    bin_total = np.bincount(binids, minlength=n_bins)
    nonzero = bin_total > 0
    bin_probs = np.bincount(binids, weights=y_prob, minlength=n_bins)[nonzero] / bin_total[nonzero]
    bin_acc = np.bincount(binids, weights=y_true, minlength=n_bins)[nonzero] / bin_total[nonzero]
    return np.sum(np.abs(bin_probs - bin_acc) * bin_total[nonzero]) / len(y_true)

df = pd.read_csv("unified_paradigm_results.csv")
# EMC_Final in this csv was p_bayes * log1p(stability) / (pr**0.5)
raw = df['EMC_Final'].values
probs = scipy.special.expit(15 * raw - 2.5)
ece = compute_ece(df['label'].values, probs)
print(f"Calibrated EMC ECE: {ece:.4f}")
