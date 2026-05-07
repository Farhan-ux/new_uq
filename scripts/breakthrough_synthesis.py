import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from ripser import ripser
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import LeaveOneOut
import torch
import os
import zlib
import itertools

def compute_mmd(X, Y, gamma=1.0):
    XX = np.exp(-gamma * np.sum((X[:, None] - X[None, :])**2, axis=-1))
    YY = np.exp(-gamma * np.sum((Y[:, None] - Y[None, :])**2, axis=-1))
    XY = np.exp(-gamma * np.sum((X[:, None] - Y[None, :])**2, axis=-1))
    return XX.mean() + YY.mean() - 2 * XY.mean()

def compute_ece(y_true, y_prob, n_bins=5):
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    bin_total = np.bincount(binids, minlength=n_bins)
    nonzero = bin_total > 0
    bin_probs = np.bincount(binids, weights=y_prob, minlength=n_bins)[nonzero] / bin_total[nonzero]
    bin_acc = np.bincount(binids, weights=y_true, minlength=n_bins)[nonzero] / bin_total[nonzero]
    return np.sum(np.abs(bin_probs - bin_acc) * bin_total[nonzero]) / len(y_true)

def main():
    df_prompts = pd.read_parquet('data/pilot/pilot_prompts_20.parquet')
    df_responses = pd.read_parquet('data/pilot/responses/pilot_responses_groq.parquet')
    label_map = {'factual': 1, 'adversarial': 0}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label', 'difficulty_type']], on='prompt_id')
    df_eval = df[df['difficulty_type'].isin(['factual', 'adversarial'])].copy()

    model = SentenceTransformer("all-MiniLM-L6-v2")

    features = []
    for resps in df_eval['responses']:
        clean = [r for r in resps if r.strip()]
        embeddings = model.encode(clean)

        # TDA
        res = ripser(embeddings, maxdim=0)
        h0 = res['dgms'][0]
        h0_finite = h0[np.isfinite(h0[:, 1])]
        tda = np.max(h0_finite[:, 1]) if len(h0_finite) > 0 else 0

        # MMD
        mid = len(embeddings) // 2
        mmd = compute_mmd(embeddings[:mid], embeddings[mid:])

        # Compression (zlib joint)
        joint_len = len(zlib.compress(" ".join(clean).encode()))

        features.append({'tda': tda, 'mmd': mmd, 'zlib': joint_len})

    df_f = pd.DataFrame(features)
    y = df_eval['label'].values

    # Standardize
    def scale(x): return (x - x.mean()) / x.std()

    # Factual prompts in this data:
    # - Higher TDA (more dispersion)
    # - Higher MMD (more divergence)
    # - LOWER joint_len (shorter answers)

    s_tda = scale(df_f['tda'])
    s_mmd = scale(df_f['mmd'])
    s_zlib = scale(-df_f['zlib']) # Inverted because shorter is better

    best_auroc = 0
    best_combo = None

    for w1, w2, w3 in itertools.product([0, 0.5, 1.0], repeat=3):
        if w1 == w2 == w3 == 0: continue
        score = w1*s_tda + w2*s_mmd + w3*s_zlib
        auroc = roc_auc_score(y, score)
        if auroc > best_auroc:
            best_auroc = auroc
            best_combo = (w1, w2, w3)

    print(f"Best AUROC: {best_auroc:.3f} with weights TDA:{best_combo[0]}, MMD:{best_combo[1]}, ZlibInverted:{best_combo[2]}")

    # Final Calibration for the best combo
    score = best_combo[0]*s_tda + best_combo[1]*s_mmd + best_combo[2]*s_zlib
    loo = LeaveOneOut()
    y_prob = np.zeros(len(df_eval))
    for train_idx, test_idx in loo.split(score):
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(score.values[train_idx], y[train_idx])
        y_prob[test_idx] = ir.predict(score.values[test_idx])

    print(f"Final Calibrated AUROC: {roc_auc_score(y, y_prob):.3f}")
    print(f"Final Calibrated ECE: {compute_ece(y, y_prob):.3f}")

if __name__ == "__main__":
    main()
