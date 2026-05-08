import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from ripser import ripser
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

class TSC_Engine:
    def __init__(self, device="cpu"):
        self.device = device
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def compute_tsc(self, resps):
        clean = [r for r in resps if r.strip()]
        if len(clean) < 4: return None
        embs = self.embed_model.encode(clean)

        # 1. Dimensionality
        n_comp = min(len(clean) - 1, 10)
        pca = PCA(n_components=n_comp)
        emb_reduced = pca.fit_transform(embs)
        evs = pca.explained_variance_
        pr = (np.sum(evs)**2) / (np.sum(evs**2) + 1e-9)

        # 2. H0 Persistence
        res_tda = ripser(emb_reduced, maxdim=0)
        h0 = res_tda['dgms'][0]
        h0_max = np.max(h0[np.isfinite(h0[:, 1])][:, 1]) if len(h0[np.isfinite(h0[:, 1])]) > 0 else 0

        # 3. Local support (Top-3 Cosine Similarity)
        cos_sims = cosine_similarity(embs)
        top3_sim = np.mean(np.sort(cos_sims[0])[-4:-1])

        # TSC Formula
        # Factual answers are supported (top3_sim), persistent (h0_max), and low-dim (1/pr^2)
        score = (h0_max * top3_sim) / (pr**2 + 1e-9)

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
    engine = TSC_Engine(device=device)
    scores, labels, models = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        s = engine.compute_tsc(row['responses'])
        if s is not None:
            scores.append(s); labels.append(row['label']); models.append(row['model'])

    df_res = pd.DataFrame({'label': labels, 'score': scores, 'model': models})
    print(f"\nOverall TSC AUROC: {roc_auc_score(labels, scores):.4f}")
    for m in df_res['model'].unique():
        print(f"Model {m} AUROC: {roc_auc_score(df_res[df_res['model']==m]['label'], df_res[df_res['model']==m]['score']):.4f}")

if __name__ == "__main__":
    run()
