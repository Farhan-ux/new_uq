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

class AMC_V6:
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

        # 2. Persistence (H0 only, higher max dim is slow)
        res_tda = ripser(emb_reduced, maxdim=0)
        h0 = res_tda['dgms'][0]
        h0_max = np.max(h0[np.isfinite(h0[:, 1])][:, 1]) if len(h0[np.isfinite(h0[:, 1])]) > 0 else 0

        # 3. Graph connectivity (Algebraic Connectivity of the semantic graph)
        # Construct graph where edges exist if cos_sim > 0.8
        cos_sims = cosine_similarity(embs)
        adj = (cos_sims > 0.8).astype(float)
        deg = np.diag(np.sum(adj, axis=1))
        laplacian = deg - adj
        vals = np.linalg.eigvalsh(laplacian)
        # Fiedler value (second smallest eigenvalue) represents connectivity
        fiedler = vals[1] if len(vals) > 1 else 0

        # 4. Centrality
        dist_matrix = euclidean_distances(embs)
        sigma = np.median(dist_matrix[np.triu_indices(len(clean), k=1)])
        if sigma == 0: sigma = 0.5
        gamma = np.mean(np.exp(-(dist_matrix**2) / (2 * sigma**2)), axis=1)
        gamma_rel = gamma[0] / (np.median(gamma) + 1e-9)

        # AMC V6: Algebraic Connectivity * Persistence * Centrality / Dimension
        score = (fiedler * h0_max * gamma_rel) / (pr + 1.0)

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
    amc = AMC_V6(device=device)
    scores, labels, models = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        s = amc.compute_metrics(row['responses'])
        if s is not None:
            scores.append(s); labels.append(row['label']); models.append(row['model'])

    df_res = pd.DataFrame({'label': labels, 'score': scores, 'model': models})
    print(f"\nOverall AMC V6 AUROC: {roc_auc_score(labels, scores):.4f}")
    for m in df_res['model'].unique():
        print(f"Model {m} AUROC: {roc_auc_score(df_res[df_res['model']==m]['label'], df_res[df_res['model']==m]['score']):.4f}")

if __name__ == "__main__":
    run()
