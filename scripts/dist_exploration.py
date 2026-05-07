import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import LeaveOneOut
import torch
import os

def compute_mmd(X, Y, kernel='rbf', gamma=None):
    """
    Computes Maximum Mean Discrepancy (MMD) between two sets of samples.
    For internal divergence, we compare a subset of samples or distance to centroid.
    """
    if kernel == 'rbf':
        if gamma is None:
            gamma = 1.0 / X.shape[1]

        def rbf_kernel(A, B):
            sq_dist = np.sum(A**2, axis=1).reshape(-1, 1) + np.sum(B**2, axis=1) - 2 * np.dot(A, B.T)
            return np.exp(-gamma * sq_dist)

        XX = rbf_kernel(X, X)
        YY = rbf_kernel(Y, Y)
        XY = rbf_kernel(X, Y)
        return XX.mean() + YY.mean() - 2 * XY.mean()
    return 0

def main():
    df_prompts = pd.read_parquet('data/pilot/pilot_prompts_20.parquet')
    df_responses = pd.read_parquet('data/pilot/responses/pilot_responses_groq.parquet')

    label_map = {'factual': 1, 'adversarial': 0}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label', 'difficulty_type']], on='prompt_id')
    df_eval = df[df['difficulty_type'].isin(['factual', 'adversarial'])].copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    dist_features = []
    for resps in df_eval['responses']:
        clean = [r for r in resps if r.strip()]
        if len(clean) < 4:
            dist_features.append({'internal_mmd': 0, 'avg_dist_to_centroid': 0})
            continue

        embeddings = model.encode(clean)

        # Split responses in two sets to compute MMD (internal diversity)
        mid = len(embeddings) // 2
        mmd = compute_mmd(embeddings[:mid], embeddings[mid:])

        # Average distance to centroid
        centroid = np.mean(embeddings, axis=0)
        dists = np.linalg.norm(embeddings - centroid, axis=1)
        avg_dist = np.mean(dists)

        dist_features.append({
            'internal_mmd': mmd,
            'avg_dist_to_centroid': avg_dist
        })

    feats_df = pd.DataFrame(dist_features)
    y = df_eval['label'].values

    results = []
    for col in feats_df.columns:
        X = feats_df[col].values
        results.append({'method': f"{col}_orig", 'auroc': roc_auc_score(y, -X)})
        results.append({'method': f"{col}_inv", 'auroc': roc_auc_score(y, X)})

    print(pd.DataFrame(results).sort_values('auroc', ascending=False).head(10))

if __name__ == "__main__":
    main()
