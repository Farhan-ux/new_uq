import pandas as pd
import numpy as np
import torch
import zlib
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from ripser import ripser
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def get_ncd(s1, s2):
    try:
        b1, b2 = s1.encode(), s2.encode()
        c1, c2 = len(zlib.compress(b1)), len(zlib.compress(b2))
        c12 = len(zlib.compress(b1 + b2))
        return (c12 - min(c1, c2)) / max(c1, c2)
    except: return 1.0

class WinningMethod:
    def __init__(self, device="cpu"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    def compute_all(self, resps):
        clean = [r for r in resps if r.strip()]
        if len(clean) < 3: return None

        embeddings = self.model.encode(clean)

        # PCA
        n_comp = min(len(clean) - 1, 10)
        pca = PCA(n_components=n_comp)
        emb_reduced = pca.fit_transform(embeddings)
        evr = pca.explained_variance_ratio_
        # Intrinsic dimensionality proxy
        id_proxy = np.sum(evr > 0.05)

        dist_matrix = euclidean_distances(embeddings)
        triu_idx = np.triu_indices(len(clean), k=1)
        sigma = np.median(dist_matrix[triu_idx])
        if sigma == 0: sigma = 0.5

        res_tda = ripser(emb_reduced, maxdim=0)
        h0 = res_tda['dgms'][0]
        h_max = np.max(h0[np.isfinite(h0[:, 1]), 1]) if len(h0) > 0 else 0

        kernel_matrix = np.exp(-(dist_matrix**2) / (2 * sigma**2))
        gammas = np.mean(kernel_matrix, axis=1)
        gamma_r1 = gammas[0]
        gamma_median = np.median(gammas)
        gamma_ratio = gamma_r1 / (gamma_median + 1e-9)

        ncds = [get_ncd(clean[0], clean[i]) for i in range(1, len(clean))]
        mean_ncd = np.mean(ncds)
        ncd_sim = 1 - mean_ncd

        return {
            'h_max': h_max,
            'gamma_ratio': gamma_ratio,
            'ncd_sim': ncd_sim,
            'id_proxy': id_proxy,
            'cos_sim': np.mean(cosine_similarity(embeddings[0:1], embeddings[1:])[0])
        }

def run_winning_search():
    df_responses = pd.read_parquet('data/pilot/responses_100/responses.parquet')
    df_prompts = pd.read_parquet('data/pilot/pilot_100_prompts.parquet')
    exclude_model = 'llama-3.3-70b-versatile'
    df_responses = df_responses[df_responses['model'] != exclude_model].copy()

    label_map = {'factual': 1, 'adversarial': 0, 'ambiguous': -1}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label']], on='prompt_id')

    engine = WinningMethod(device="cpu")

    data = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Winning Search"):
        f = engine.compute_all(row['responses'])
        if f:
            f['label'] = row['label']
            f['model'] = row['model']
            data.append(f)

    df_f = pd.DataFrame(data)
    df_eval = df_f[df_f['label'].isin([0, 1])]

    # Hypothesize: SAME (Semantic-Algorithmic Manifold Evidence)
    # Factor 1: Topological Structure (tanh(H_max))
    # Factor 2: Local Evidence (Gamma Ratio)
    # Factor 3: Algorithmic Soundness (NCD Similarity)

    df_eval['SAME_v1'] = np.tanh(df_eval['h_max']) * df_eval['gamma_ratio'] * df_eval['ncd_sim']
    df_eval['SAME_v2'] = (df_eval['h_max'] * df_eval['gamma_ratio']) / (df_eval['id_proxy'] + 1)
    df_eval['SAME_v3'] = df_eval['gamma_ratio'] * (df_eval['h_max'] + df_eval['ncd_sim'])

    print("\n--- Winning Search Results ---")
    for col in ['SAME_v1', 'SAME_v2', 'SAME_v3']:
        auc = roc_auc_score(df_eval['label'], df_eval[col])
        print(f"Method {col}: Overall AUROC = {auc:.4f}")

    for model in df_eval['model'].unique():
        print(f"\nModel: {model}")
        m_df = df_eval[df_eval['model'] == model]
        for col in ['SAME_v1', 'SAME_v2', 'SAME_v3']:
            print(f"  {col}: {roc_auc_score(m_df['label'], m_df[col]):.4f}")

if __name__ == "__main__":
    run_winning_search()
