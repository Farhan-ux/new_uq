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

def get_lexical_redundancy(resps):
    ind_sizes = [len(zlib.compress(r.encode())) for r in resps]
    joint_size = len(zlib.compress(" ".join(resps).encode()))
    return np.sum(ind_sizes) / (joint_size + 1e-9)

class AMC_V20:
    def __init__(self, device="cpu"):
        self.device = device
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def compute_score(self, resps):
        clean = [r for r in resps if r.strip()]
        n = len(clean)
        if n < 4: return None
        embs = self.embed_model.encode(clean)
        pca = PCA(n_components=min(n-1, 10))
        emb_red = pca.fit_transform(embs)
        res_tda = ripser(emb_red, maxdim=0)
        h0_max = np.max(res_tda['dgms'][0][np.isfinite(res_tda['dgms'][0][:, 1])][:, 1]) if len(res_tda['dgms'][0][np.isfinite(res_tda['dgms'][0][:, 1])]) > 0 else 0

        evs = pca.explained_variance_
        stable_rank = np.sum(evs) / (evs[0] + 1e-9)

        cos_sims = cosine_similarity(embs)
        gamma_rel = (np.sum(cos_sims[0]) - 1.0) / (np.median(np.sum(cos_sims, axis=1) - 1.0) + 1e-9)

        redundancy = get_lexical_redundancy(clean)

        # Synergy Formula: Power scaling for non-linear interaction
        # S = (Topo * Centrality)^0.75 * (Redundancy / Rank)^0.5
        score = ((h0_max * gamma_rel)**0.75) * ((redundancy / (stable_rank + 1e-9))**0.5)

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
    engine = AMC_V20(device=device)
    scores, labels, models = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        s = engine.compute_score(row['responses'])
        if s is not None:
            scores.append(s); labels.append(row['label']); models.append(row['model'])

    auc = roc_auc_score(labels, scores)
    print(f"\nAMC v20 AUROC: {auc:.4f}")
    df_res = pd.DataFrame({'label': labels, 'score': scores, 'model': models})
    for m in df_res['model'].unique():
        print(f"Model {m} AUROC: {roc_auc_score(df_res[df_res['model']==m]['label'], df_res[df_res['model']==m]['score']):.4f}")

if __name__ == "__main__":
    run()
