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

class AMEMethod:
    def __init__(self, device="cpu"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    def compute_scores(self, resps):
        clean = [r for r in resps if r.strip()]
        if len(clean) < 3: return None

        embeddings = self.model.encode(clean)

        # 1. iTME Component
        n_comp = min(len(clean) - 1, 10)
        pca = PCA(n_components=0.95 if len(clean) > 5 else n_comp)
        try:
            emb_reduced = pca.fit_transform(embeddings)
        except:
            emb_reduced = embeddings

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

        itme_score = (gamma_r1 / (gamma_median + 1e-9)) * np.tanh(h_max)

        # 2. Algorithmic Component (NCD)
        ncds = [get_ncd(clean[0], clean[i]) for i in range(1, len(clean))]
        mean_ncd = np.mean(ncds)

        # 3. Semantic Density (Cosine)
        cos_sims = cosine_similarity(embeddings[0:1], embeddings[1:])[0]
        mean_cos = np.mean(cos_sims)

        return {
            'itme': itme_score,
            'ncd_sim': 1 - mean_ncd,
            'cos_sim': mean_cos,
            'h_max': h_max
        }

def run_exploration():
    df_responses = pd.read_parquet('data/pilot/responses_100/responses.parquet')
    df_prompts = pd.read_parquet('data/pilot/pilot_100_prompts.parquet')
    exclude_model = 'llama-3.3-70b-versatile'
    df_responses = df_responses[df_responses['model'] != exclude_model].copy()

    label_map = {'factual': 1, 'adversarial': 0, 'ambiguous': -1}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label']], on='prompt_id')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = AMEMethod(device=device)

    features = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Exploring"):
        f = engine.compute_scores(row['responses'])
        if f:
            f['label'] = row['label']
            f['model'] = row['model']
            features.append(f)

    feat_df = pd.DataFrame(features)
    eval_df = feat_df[feat_df['label'].isin([0, 1])]

    print("\n--- Exploration Results (AUROC) ---")

    # Test combinations
    # 1. iTME
    # 2. iTME * cos_sim
    # 3. iTME * ncd_sim
    # 4. h_max * cos_sim * gamma_ratio (essentially iTME but weight h_max more?)

    eval_df['itme_cos'] = eval_df['itme'] * eval_df['cos_sim']
    eval_df['itme_ncd'] = eval_df['itme'] * eval_df['ncd_sim']
    eval_df['topo_alg'] = np.tanh(eval_df['h_max']) * eval_df['ncd_sim']

    for col in ['itme', 'itme_cos', 'itme_ncd', 'topo_alg']:
        auc = roc_auc_score(eval_df['label'], eval_df[col])
        print(f"Method {col}: Overall AUROC = {auc:.4f}")

    for model in eval_df['model'].unique():
        m_df = eval_df[eval_df['model'] == model]
        print(f"\nModel: {model}")
        for col in ['itme', 'itme_cos', 'itme_ncd', 'topo_alg']:
            auc = roc_auc_score(m_df['label'], m_df[col])
            print(f"  {col}: {auc:.4f}")

if __name__ == "__main__":
    run_exploration()
