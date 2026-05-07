import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from ripser import ripser
import zlib
import itertools
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
import torch
import os
from tqdm import tqdm

def compute_ece(y_true, y_prob, n_bins=5):
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    bin_total = np.bincount(binids, minlength=n_bins)
    nonzero = bin_total > 0
    bin_probs = np.bincount(binids, weights=y_prob, minlength=n_bins)[nonzero] / bin_total[nonzero]
    bin_acc = np.bincount(binids, weights=y_true, minlength=n_bins)[nonzero] / bin_total[nonzero]
    return np.sum(np.abs(bin_probs - bin_acc) * bin_total[nonzero]) / len(y_true)

def get_ncd(s1, s2):
    try:
        b1, b2 = s1.encode(), s2.encode()
        c1, c2 = len(zlib.compress(b1)), len(zlib.compress(b2))
        c12 = len(zlib.compress(b1 + b2))
        return (c12 - min(c1, c2)) / max(c1, c2)
    except: return 1.0

def main():
    df_prompts = pd.read_parquet('data/pilot/pilot_prompts_20.parquet')
    df_responses = pd.read_parquet('data/pilot/responses/pilot_responses_groq.parquet')

    label_map = {'factual': 1, 'adversarial': 0}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label', 'difficulty_type']], on='prompt_id')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    results_list = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="SSD Feature Extraction"):
        resps = [r for r in row['responses'] if r.strip()]
        if len(resps) < 2:
            results_list.append({'tda': 0.0, 'ncd_jitter': 0.0})
            continue

        embeddings = model.encode(resps)
        res_tda = ripser(embeddings, maxdim=0)
        h0 = res_tda['dgms'][0]
        h0_finite = h0[np.isfinite(h0[:, 1])]
        tda_feat = np.max(h0_finite[:, 1]) if len(h0_finite) > 0 else 0

        pairs = list(itertools.combinations(resps, 2))
        ncd_vals = [get_ncd(p[0], p[1]) for p in pairs]
        ncd_feat = np.std(ncd_vals)
        results_list.append({'tda': tda_feat, 'ncd_jitter': ncd_feat})

    df_feats = pd.DataFrame(results_list)
    def zscale(x): return (x - x.mean()) / x.std()
    df['ssd_raw'] = (zscale(df_feats['tda']) + zscale(df_feats['ncd_jitter'])) / 2.0

    df_eval = df[df['difficulty_type'].isin(['factual', 'adversarial'])].copy()
    X = df_eval['ssd_raw'].values
    y = df_eval['label'].values

    loo = LeaveOneOut()
    y_prob = np.zeros(len(df_eval))
    for train_idx, test_idx in loo.split(X):
        lr = LogisticRegression()
        lr.fit(X[train_idx].reshape(-1, 1), y[train_idx])
        y_prob[test_idx] = lr.predict_proba(X[test_idx].reshape(-1, 1))[:, 1]

    auroc = roc_auc_score(y, y_prob)
    ece = compute_ece(y, y_prob)
    print(f"SSD LOO AUROC: {auroc:.3f}, ECE: {ece:.3f}")

    # We apply the LOO-calibrated probabilities to the full set
    # For simplicity, we'll re-fit on full train for the final scoring but the summary will use LOO metrics
    lr_final = LogisticRegression()
    lr_final.fit(X.reshape(-1, 1), y)
    df['confidence_score'] = lr_final.predict_proba(df['ssd_raw'].values.reshape(-1, 1))[:, 1]
    df['uncertainty_score'] = 1.0 - df['confidence_score']
    df['method'] = 'Structural_Semantic_Dispersion'

    df[['prompt_id', 'model', 'method', 'uncertainty_score', 'confidence_score', 'n_generated']].rename(
        columns={'n_generated': 'n_responses_used'}
    ).to_parquet('experiments/pilot_uq_benchmark/ssd_scores.parquet')

if __name__ == "__main__":
    main()
