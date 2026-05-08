import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from ripser import ripser
import scipy.special
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

class AMC_V3:
    def __init__(self, device="cpu"):
        self.device = device
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def compute_metrics(self, resps):
        clean = [r for r in resps if r.strip()]
        if len(clean) < 4: return None
        embs = self.embed_model.encode(clean)

        # 1. Manifold Dimensionality (Participation Ratio)
        # Truth manifolds should be highly compressed
        n_comp = min(len(clean) - 1, 10)
        pca = PCA(n_components=n_comp)
        emb_reduced = pca.fit_transform(embs)
        evs = pca.explained_variance_
        pr = (np.sum(evs)**2) / (np.sum(evs**2) + 1e-9)

        # 2. Persistence (H0 & H1)
        res_tda = ripser(emb_reduced, maxdim=1)
        h0 = res_tda['dgms'][0]
        h0_max = np.max(h0[np.isfinite(h0[:, 1])][:, 1]) if len(h0[np.isfinite(h0[:, 1])]) > 0 else 0
        h1 = res_tda['dgms'][1]
        h1_max = np.max(h1[:, 1] - h1[:, 0]) if len(h1) > 0 else 0

        # 3. Geometric Centrality
        dist_matrix = euclidean_distances(embs)
        sigma = np.median(dist_matrix[np.triu_indices(len(clean), k=1)])
        if sigma == 0: sigma = 0.5
        gamma = np.mean(np.exp(-(dist_matrix**2) / (2 * sigma**2)), axis=1)
        gamma_rel = gamma[0] / (np.median(gamma) + 1e-9)

        # 4. Local Consistency (Top-k Density)
        # Average similarity to top 3 neighbors
        cos_sims = cosine_similarity(embs)
        top3_sim = np.mean(np.sort(cos_sims[0])[-4:-1])

        # AMC V3: Combine persistence, spectral compression and local density
        # We find that h0_max / pr is a strong structural signal.
        # h1_max is a strong hallucination indicator.
        # gamma_rel * top3_sim is the local support.

        structural_signal = h0_max / (pr * (1.0 + h1_max))
        support_signal = gamma_rel * top3_sim

        return structural_signal * support_signal

def run():
    df_responses = pd.read_parquet('data/pilot/responses_100/responses.parquet')
    df_prompts = pd.read_parquet('data/pilot/pilot_100_prompts.parquet')
    df_responses = df_responses[df_responses['model'] != 'llama-3.3-70b-versatile'].copy()
    label_map = {'factual': 1, 'adversarial': 0}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label']], on='prompt_id')
    df = df[df['label'].isin([0, 1])].copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amc = AMC_V3(device=device)
    scores = []
    labels = []
    models = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        s = amc.compute_metrics(row['responses'])
        if s is not None:
            scores.append(s)
            labels.append(row['label'])
            models.append(row['model'])

    df_res = pd.DataFrame({'label': labels, 'score': scores, 'model': models})
    print(f"\nOverall AMC V3 AUROC: {roc_auc_score(labels, scores):.4f}")
    for m in df_res['model'].unique():
        m_auc = roc_auc_score(df_res[df_res['model']==m]['label'], df_res[df_res['model']==m]['score'])
        print(f"Model {m} AUROC: {m_auc:.4f}")

if __name__ == "__main__":
    run()
