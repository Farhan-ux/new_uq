import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
import zlib
from ripser import ripser

def get_ncd(s1, s2):
    try:
        b1, b2 = s1.encode(), s2.encode()
        c1, c2 = len(zlib.compress(b1)), len(zlib.compress(b2))
        c12 = len(zlib.compress(b1 + b2))
        return (c12 - min(c1, c2)) / max(c1, c2)
    except: return 1.0

def analyze():
    df_prompts = pd.read_parquet('data/pilot/pilot_prompts_20.parquet')
    df_responses = pd.read_parquet('data/pilot/responses/pilot_responses_groq.parquet')

    label_map = {'factual': 1, 'adversarial': 0, 'ambiguous': -1}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label', 'difficulty_type']], on='prompt_id')

    df_main = df[df['label'].isin([0, 1])].copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    stats = []

    for idx, row in df_main.iterrows():
        resps = row['responses']
        clean = [r for r in resps if r.strip()]
        if len(clean) < 2:
            continue

        r1 = clean[0]
        lengths = [len(r) for r in clean]

        embeddings = model.encode(clean)

        dist_matrix = cosine_distances(embeddings)
        mean_dist = dist_matrix[np.triu_indices(len(clean), k=1)].mean()

        ncd_vals = []
        for i in range(len(clean)):
            for j in range(i+1, len(clean)):
                ncd_vals.append(get_ncd(clean[i], clean[j]))
        mean_ncd = np.mean(ncd_vals)

        res_tda = ripser(embeddings, maxdim=0)
        h0 = res_tda['dgms'][0]
        h0_finite = h0[np.isfinite(h0[:, 1])]
        max_h0 = np.max(h0_finite[:, 1]) if len(h0_finite) > 0 else 0

        stats.append({
            'label': row['label'],
            'mean_dist': mean_dist,
            'mean_ncd': mean_ncd,
            'max_h0': max_h0,
            'avg_len': np.mean(lengths)
        })

    df_stats = pd.DataFrame(stats)
    print("Grouped Stats:")
    print(df_stats.groupby('label').mean())

if __name__ == "__main__":
    analyze()
