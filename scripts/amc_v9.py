import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from ripser import ripser
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

class TSM_Engine:
    def __init__(self, device="cpu"):
        self.device = device
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def compute_tsm(self, resps):
        clean = [r for r in resps if r.strip()]
        if len(clean) < 4: return None
        embs = self.embed_model.encode(clean)

        # 1. Topological Stability
        pca = PCA(n_components=min(len(clean)-1, 10))
        emb_red = pca.fit_transform(embs)
        res_tda = ripser(emb_red, maxdim=1)
        h0_max = np.max(res_tda['dgms'][0][np.isfinite(res_tda['dgms'][0][:, 1])][:, 1]) if len(res_tda['dgms'][0][np.isfinite(res_tda['dgms'][0][:, 1])]) > 0 else 0
        h1_max = np.max(res_tda['dgms'][1][:, 1] - res_tda['dgms'][1][:, 0]) if len(res_tda['dgms'][1]) > 0 else 0
        stability = h0_max / (1.0 + h1_max)

        # 2. Spectral Compression
        evs = pca.explained_variance_
        pr = (np.sum(evs)**2) / (np.sum(evs**2) + 1e-9)

        # 3. Medoid Consensus
        # Medoid is the response with the highest sum of similarities to all others
        cos_sims = cosine_similarity(embs)
        medoid_idx = np.argmax(np.sum(cos_sims, axis=1))
        # Ground r1 (the first response) against the medoid
        medoid_support = cos_sims[0, medoid_idx]

        # 4. TSM Formula
        # Factual truth is stable (stability), compressed (1/pr), and grounded (medoid_support)
        score = (stability * medoid_support) / (pr + 1e-9)

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
    engine = TSM_Engine(device=device)
    scores, labels, models = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        s = engine.compute_tsm(row['responses'])
        if s is not None:
            scores.append(s); labels.append(row['label']); models.append(row['model'])

    df_res = pd.DataFrame({'label': labels, 'score': scores, 'model': models})
    print(f"\nOverall TSM AUROC: {roc_auc_score(labels, scores):.4f}")
    for m in df_res['model'].unique():
        print(f"Model {m} AUROC: {roc_auc_score(df_res[df_res['model']==m]['label'], df_res[df_res['model']==m]['score']):.4f}")

if __name__ == "__main__":
    run()
