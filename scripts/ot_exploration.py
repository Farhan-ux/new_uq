import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
from scipy.stats import wasserstein_distance
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

    ot_features = []
    for resps in df_eval['responses']:
        clean = [r for r in resps if r.strip()]
        if len(clean) < 4:
            ot_features.append({'avg_wasserstein': 0})
            continue

        embeddings = model.encode(clean)

        # Compute pairwise 1D Wasserstein distances (projection to 1st principal component)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        proj = pca.fit_transform(embeddings).flatten()

        # Split and compare distributions
        mid = len(proj) // 2
        w_dist = wasserstein_distance(proj[:mid], proj[mid:])

        ot_features.append({
            'wasserstein_proj': w_dist
        })

    feats_df = pd.DataFrame(ot_features)
    y = df_eval['label'].values

    results = []
    for col in feats_df.columns:
        X = feats_df[col].values
        results.append({'method': f"{col}_orig", 'auroc': roc_auc_score(y, -X)})
        results.append({'method': f"{col}_inv", 'auroc': roc_auc_score(y, X)})

    print(pd.DataFrame(results).sort_values('auroc', ascending=False).head(10))

if __name__ == "__main__":
    main()
