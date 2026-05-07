import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from ripser import ripser
from persim import plot_diagrams
from sklearn.metrics import roc_auc_score
import torch
import os

def main():
    df_prompts = pd.read_parquet('data/pilot/pilot_prompts_20.parquet')
    df_responses = pd.read_parquet('data/pilot/responses/pilot_responses_groq.parquet')

    label_map = {'factual': 1, 'adversarial': 0}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label', 'difficulty_type']], on='prompt_id')
    df_eval = df[df['difficulty_type'].isin(['factual', 'adversarial'])].copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    tda_features = []
    for resps in df_eval['responses']:
        clean = [r for r in resps if r.strip()]
        if len(clean) < 2:
            tda_features.append({'b0_persistence': 0, 'b1_persistence': 0, 'max_persistence': 0})
            continue

        embeddings = model.encode(clean)
        # Compute persistence homology
        # ripser returns a dict with 'dgms' containing list of persistence diagrams
        result = ripser(embeddings)
        dgms = result['dgms']

        # Feature 1: Total persistence of H0 (clusters)
        # H0 diagram: [0, death] for each component.
        # Persistence = death
        h0 = dgms[0]
        # remove the point with infinity death
        h0_finite = h0[np.isfinite(h0[:, 1])]
        b0_pers = np.sum(h0_finite[:, 1]) if len(h0_finite) > 0 else 0

        # Feature 2: Max persistence in H1 (loops)
        h1 = dgms[1]
        b1_pers = np.max(h1[:, 1] - h1[:, 0]) if len(h1) > 0 else 0

        tda_features.append({
            'b0_persistence': b0_pers,
            'b1_persistence': b1_pers,
            'max_h0_death': np.max(h0_finite[:, 1]) if len(h0_finite) > 0 else 0
        })

    feats_df = pd.DataFrame(tda_features)
    y = df_eval['label'].values

    results = []
    for col in feats_df.columns:
        X = feats_df[col].values
        # Test original and inverted
        results.append({'method': f"{col}_orig", 'auroc': roc_auc_score(y, -X)})
        results.append({'method': f"{col}_inv", 'auroc': roc_auc_score(y, X)})

    print(pd.DataFrame(results).sort_values('auroc', ascending=False).head(10))

if __name__ == "__main__":
    main()
