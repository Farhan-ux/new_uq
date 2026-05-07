import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from ripser import ripser
from sklearn.metrics import roc_auc_score
import torch
import os

def compute_mmd(X, Y, gamma=1.0):
    XX = np.exp(-gamma * np.sum((X[:, None] - X[None, :])**2, axis=-1))
    YY = np.exp(-gamma * np.sum((Y[:, None] - Y[None, :])**2, axis=-1))
    XY = np.exp(-gamma * np.sum((X[:, None] - Y[None, :])**2, axis=-1))
    return XX.mean() + YY.mean() - 2 * XY.mean()

def main():
    df_prompts = pd.read_parquet('data/pilot/pilot_prompts_20.parquet')
    df_responses = pd.read_parquet('data/pilot/responses/pilot_responses_groq.parquet')
    label_map = {'factual': 1, 'adversarial': 0}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label', 'difficulty_type']], on='prompt_id')
    df_eval = df[df['difficulty_type'].isin(['factual', 'adversarial'])].copy()

    model = SentenceTransformer("all-MiniLM-L6-v2")

    tda_feats = []
    mmd_feats = []
    for resps in df_eval['responses']:
        clean = [r for r in resps if r.strip()]
        embeddings = model.encode(clean)

        # TDA
        res = ripser(embeddings, maxdim=0)
        h0 = res['dgms'][0]
        h0_finite = h0[np.isfinite(h0[:, 1])]
        tda_feats.append(np.max(h0_finite[:, 1]) if len(h0_finite) > 0 else 0)

        # MMD
        mid = len(embeddings) // 2
        mmd_feats.append(compute_mmd(embeddings[:mid], embeddings[mid:]))

    y = df_eval['label'].values

    def norm(x): return (x - np.mean(x)) / np.std(x)

    s1 = norm(np.array(tda_feats))
    s2 = norm(np.array(mmd_feats))

    print(f"TDA AUROC: {roc_auc_score(y, s1):.3f}")
    print(f"MMD AUROC: {roc_auc_score(y, s2):.3f}")
    print(f"Ensemble AUROC: {roc_auc_score(y, s1 + s2):.3f}")

if __name__ == "__main__":
    main()
