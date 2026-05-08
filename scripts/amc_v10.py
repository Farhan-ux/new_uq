import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from ripser import ripser
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

class UMC_Engine:
    def __init__(self, device="cpu"):
        self.device = device
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def compute_umc(self, resps):
        clean = [r for r in resps if r.strip()]
        if len(clean) < 4: return None
        embs = self.embed_model.encode(clean)

        # 1. Manifold Dimensionality (Participation Ratio)
        pca = PCA(n_components=min(len(clean)-1, 10))
        emb_red = pca.fit_transform(embs)
        evs = pca.explained_variance_
        pr = (np.sum(evs)**2) / (np.sum(evs**2) + 1e-9)

        # 2. Persistence Stability (H0 & H1)
        res_tda = ripser(emb_red, maxdim=1)
        h0 = res_tda['dgms'][0]
        h0_max = np.max(h0[np.isfinite(h0[:, 1])][:, 1]) if len(h0[np.isfinite(h0[:, 1])]) > 0 else 0
        h1 = res_tda['dgms'][1]
        h1_max = np.max(h1[:, 1] - h1[:, 0]) if len(h1) > 0 else 0

        # 3. Representativeness (Distance to Medoid)
        cos_sims = cosine_similarity(embs)
        # Sum of similarities excluding self
        sim_sums = np.sum(cos_sims, axis=1) - 1.0
        medoid_idx = np.argmax(sim_sums)
        r1_medoid_sim = cos_sims[0, medoid_idx]

        # 4. Local Clusteredness
        # Similarity to 2nd and 3rd nearest neighbors
        sorted_sims = np.sort(cos_sims[0])
        local_clust = np.mean(sorted_sims[-4:-1])

        # UMC Formula:
        # Persistence scaled by 1/PR^1.5 (heavy penalty for high-dim noise)
        # and penalized by H1 confusion loops.
        # Support is the product of medoid representativeness and local clusteredness.

        structural = h0_max / ( (pr**1.5) * (1.0 + h1_max) )
        support = r1_medoid_sim * local_clust

        return structural * support

def run():
    df_responses = pd.read_parquet('data/pilot/responses_100/responses.parquet')
    df_prompts = pd.read_parquet('data/pilot/pilot_100_prompts.parquet')
    df_responses = df_responses[df_responses['model'] != 'llama-3.3-70b-versatile'].copy()
    label_map = {'factual': 1, 'adversarial': 0}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label']], on='prompt_id')
    df = df[df['label'].isin([0, 1])].copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = UMC_Engine(device=device)
    scores, labels, models = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        s = engine.compute_umc(row['responses'])
        if s is not None:
            scores.append(s); labels.append(row['label']); models.append(row['model'])

    print(f"\nOverall UMC AUROC: {roc_auc_score(labels, scores):.4f}")
    df_res = pd.DataFrame({'label': labels, 'score': scores, 'model': models})
    for m in df_res['model'].unique():
        print(f"Model {m} AUROC: {roc_auc_score(df_res[df_res['model']==m]['label'], df_res[df_res['model']==m]['score']):.4f}")

if __name__ == "__main__":
    run()
