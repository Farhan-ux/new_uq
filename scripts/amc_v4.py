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

class AMC_V4:
    def __init__(self, device="cpu"):
        self.device = device
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def compute_metrics(self, resps):
        clean = [r for r in resps if r.strip()]
        if len(clean) < 4: return None
        embs = self.embed_model.encode(clean)

        # 1. Manifold Dimensionality (Participation Ratio)
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

        # 3. Dynamic Local Density (Inverse Average K-NN Distance)
        # Using k=3
        dist_matrix = euclidean_distances(embs)
        k = 3
        knn_dists = np.sort(dist_matrix, axis=1)[:, 1:k+1]
        avg_knn_dist = np.mean(knn_dists, axis=1)
        local_density_r1 = 1.0 / (avg_knn_dist[0] + 1e-9)
        med_local_density = np.median(1.0 / (avg_knn_dist + 1e-9))
        density_rel = local_density_r1 / (med_local_density + 1e-9)

        # 4. AMC V4: Geometric-Topological Consensus
        # Signal = (Persistence / Scaling) * (Support / Variability)
        # Scale by PR to penalize high-dim scattering

        structural_signal = h0_max / (pr * (1.0 + h1_max))
        # Add a logarithmic compression to structural signal to prevent outliers
        structural_signal = np.log1p(structural_signal)

        final_score = structural_signal * density_rel

        return final_score

def run():
    df_responses = pd.read_parquet('data/pilot/responses_100/responses.parquet')
    df_prompts = pd.read_parquet('data/pilot/pilot_100_prompts.parquet')
    df_responses = df_responses[df_responses['model'] != 'llama-3.3-70b-versatile'].copy()
    label_map = {'factual': 1, 'adversarial': 0}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label']], on='prompt_id')
    df = df[df['label'].isin([0, 1])].copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amc = AMC_V4(device=device)
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
    print(f"\nOverall AMC V4 AUROC: {roc_auc_score(labels, scores):.4f}")
    for m in df_res['model'].unique():
        m_auc = roc_auc_score(df_res[df_res['model']==m]['label'], df_res[df_res['model']==m]['score'])
        print(f"Model {m} AUROC: {m_auc:.4f}")

if __name__ == "__main__":
    run()
