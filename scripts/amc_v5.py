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

class AMC_V5:
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

        # 2. Persistence Stability
        res_tda = ripser(emb_reduced, maxdim=1)
        h0 = res_tda['dgms'][0]
        h0_max = np.max(h0[np.isfinite(h0[:, 1])][:, 1]) if len(h0[np.isfinite(h0[:, 1])]) > 0 else 0
        h1 = res_tda['dgms'][1]
        h1_max = np.max(h1[:, 1] - h1[:, 0]) if len(h1) > 0 else 0

        # Stability is the ratio of meaningful structure to topological noise
        stability = h0_max / (1.0 + h1_max)

        # 3. Normalized Centrality (Z-score of r1 centrality)
        dist_matrix = euclidean_distances(embs)
        sigma = np.median(dist_matrix[np.triu_indices(len(clean), k=1)])
        if sigma == 0: sigma = 0.5
        gamma = np.mean(np.exp(-(dist_matrix**2) / (2 * sigma**2)), axis=1)
        gamma_z = (gamma[0] - np.mean(gamma)) / (np.std(gamma) + 1e-9)

        # 4. Semantic Purity (Fraction of variance in top 2 components)
        purity = np.sum(pca.explained_variance_ratio_[:2])

        # AMC V5: Stability * Centrality * Purity / Complexity
        # Logic: Factual answers are stable, prominent, pure, and low-dimensional.
        # Use exp(gamma_z) to keep it positive and amplify prominence
        score = (stability * np.exp(gamma_z) * purity) / pr

        return score

def run():
    df_responses = pd.read_parquet('data/pilot/responses_100/responses.parquet')
    df_prompts = pd.read_parquet('data/pilot/pilot_100_prompts.parquet')
    df_responses = df_responses[df_responses['model'] != 'llama-3.3-70b-versatile'].copy()
    label_map = {'factual': 1, 'adversarial': 0}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label']], on='prompt_id')
    df = df[df['label'].isin([0, 1])].copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amc = AMC_V5(device=device)
    scores, labels, models = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        s = amc.compute_metrics(row['responses'])
        if s is not None:
            scores.append(s); labels.append(row['label']); models.append(row['model'])

    df_res = pd.DataFrame({'label': labels, 'score': scores, 'model': models})
    print(f"\nOverall AMC V5 AUROC: {roc_auc_score(labels, scores):.4f}")
    for m in df_res['model'].unique():
        print(f"Model {m} AUROC: {roc_auc_score(df_res[df_res['model']==m]['label'], df_res[df_res['model']==m]['score']):.4f}")

if __name__ == "__main__":
    run()
