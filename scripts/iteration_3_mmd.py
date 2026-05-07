import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import LeaveOneOut
import torch
import os
import json

def compute_mmd(X, Y, gamma=1.0):
    XX = np.exp(-gamma * np.sum((X[:, None] - X[None, :])**2, axis=-1))
    YY = np.exp(-gamma * np.sum((Y[:, None] - Y[None, :])**2, axis=-1))
    XY = np.exp(-gamma * np.sum((X[:, None] - Y[None, :])**2, axis=-1))
    return XX.mean() + YY.mean() - 2 * XY.mean()

def main():
    # Hypothesis: Factual response distributions have higher internal variance
    # (Maximum Mean Discrepancy between subsets) than formulaic hallucinations.

    df_prompts = pd.read_parquet('data/pilot/pilot_prompts_20.parquet')
    df_responses = pd.read_parquet('data/pilot/responses/pilot_responses_groq.parquet')

    label_map = {'factual': 1, 'adversarial': 0}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label', 'difficulty_type']], on='prompt_id')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    print("Computing Internal MMD...")
    features = []
    for resps in df['responses']:
        clean = [r for r in resps if r.strip()]
        if len(clean) < 4:
            features.append(0.0)
            continue
        embeddings = model.encode(clean)
        # Split into two sets
        mid = len(embeddings) // 2
        mmd = compute_mmd(embeddings[:mid], embeddings[mid:])
        features.append(mmd)

    df['internal_mmd'] = features

    df_eval = df[df['difficulty_type'].isin(['factual', 'adversarial'])].copy()
    X = df_eval['internal_mmd'].values
    y = df_eval['label'].values

    # Calibration
    loo = LeaveOneOut()
    y_prob = np.zeros(len(df_eval))
    for train_idx, test_idx in loo.split(X):
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(X[train_idx], y[train_idx])
        y_prob[test_idx] = ir.predict(X[test_idx])

    auroc = roc_auc_score(y, y_prob)

    print(f"Iteration 3 Metrics: AUROC={auroc:.3f}")

    # Decision: AUROC 0.655 (from exploration) is good.
    # Let's see if we can do better by combining TDA and MMD.

if __name__ == "__main__":
    main()
