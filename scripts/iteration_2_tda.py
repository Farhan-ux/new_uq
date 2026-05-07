import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from ripser import ripser
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import LeaveOneOut
import torch
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

def compute_ece(y_true, y_prob, n_bins=5):
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    bin_total = np.bincount(binids, minlength=n_bins)
    nonzero = bin_total > 0
    bin_probs = np.bincount(binids, weights=y_prob, minlength=n_bins)[nonzero] / bin_total[nonzero]
    bin_acc = np.bincount(binids, weights=y_true, minlength=n_bins)[nonzero] / bin_total[nonzero]
    return np.sum(np.abs(bin_probs - bin_acc) * bin_total[nonzero]) / len(y_true)

def main():
    # Hypothesis: Factual prompts have higher semantic dispersion (max H0 persistence)
    # than confidently wrong adversarial hallucinations.

    df_prompts = pd.read_parquet('data/pilot/pilot_prompts_20.parquet')
    df_responses = pd.read_parquet('data/pilot/responses/pilot_responses_groq.parquet')

    label_map = {'factual': 1, 'adversarial': 0}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label', 'difficulty_type']], on='prompt_id')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    print("Computing TDA Persistence...")
    features = []
    for resps in df['responses']:
        clean = [r for r in resps if r.strip()]
        if len(clean) < 2:
            features.append(0.0)
            continue
        embeddings = model.encode(clean)
        # Rips filtration
        res = ripser(embeddings, maxdim=0)
        h0 = res['dgms'][0]
        h0_finite = h0[np.isfinite(h0[:, 1])]
        # Max persistence in H0 represents the distance between the two most distinct clusters
        feat = np.max(h0_finite[:, 1]) if len(h0_finite) > 0 else 0
        features.append(feat)

    df['tda_h0_max'] = features

    df_eval = df[df['difficulty_type'].isin(['factual', 'adversarial'])].copy()
    X = df_eval['tda_h0_max'].values
    y = df_eval['label'].values

    # Calibration
    loo = LeaveOneOut()
    y_prob = np.zeros(len(df_eval))
    for train_idx, test_idx in loo.split(X):
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(X[train_idx], y[train_idx])
        y_prob[test_idx] = ir.predict(X[test_idx])

    auroc = roc_auc_score(y, y_prob)
    ece = compute_ece(y, y_prob)

    print(f"Iteration 2 Metrics: AUROC={auroc:.3f}, ECE={ece:.3f}")

    # Save required results
    output_dir = 'data/pilot/uq_results'
    os.makedirs(output_dir, exist_ok=True)

    # Apply final model
    ir_final = IsotonicRegression(out_of_bounds='clip')
    ir_final.fit(X, y)
    df['confidence_score'] = ir_final.predict(df['tda_h0_max'].values)
    df['uncertainty_score'] = 1.0 - df['confidence_score']
    df['method'] = 'Topological_H0_Persistence'

    df[['prompt_id', 'model', 'method', 'uncertainty_score', 'confidence_score']].to_parquet('data/pilot/uq_results/uq_results_tda.parquet')

    # Per-method JSON
    per_prompt = {}
    for i, row in df_eval.iterrows():
        p_id, m = row['prompt_id'], row['model']
        if p_id not in per_prompt: per_prompt[p_id] = {}
        per_prompt[p_id][m] = float(y_prob[list(df_eval.index).index(i)])

    json_out = {
        "method_name": "Topological_H0_Persistence",
        "auroc": float(auroc),
        "ece": float(ece),
        "hypothesis": "Factual prompts have higher semantic dispersion (max H0 persistence) than adversarial hallucinations.",
        "per_prompt_scores": per_prompt
    }
    with open(os.path.join(output_dir, "tda_h0.json"), 'w') as f:
        json.dump(json_out, f, indent=4)

if __name__ == "__main__":
    main()
