import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from ripser import ripser
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

class TC_Engine:
    def __init__(self, device="cpu"):
        self.device = device
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def compute_tc(self, resps):
        clean = [r for r in resps if r.strip()]
        n = len(clean)
        if n < 4: return None
        embs = self.embed_model.encode(clean)

        # 1. Persistence Gap
        pca = PCA(n_components=min(n-1, 10))
        emb_red = pca.fit_transform(embs)
        res_tda = ripser(emb_red, maxdim=0)
        h0_lifetimes = np.sort(res_tda['dgms'][0][np.isfinite(res_tda['dgms'][0][:, 1])][:, 1])[::-1]

        # Dominance = ratio of largest lifetime to second largest
        l1 = h0_lifetimes[0] if len(h0_lifetimes) > 0 else 0
        l2 = h0_lifetimes[1] if len(h0_lifetimes) > 1 else 0
        dominance = l1 / (l2 + 0.1) # Soft constraint

        # 2. Spectral Compression
        evs = pca.explained_variance_
        pr = (np.sum(evs)**2) / (np.sum(evs**2) + 1e-9)

        # 3. Representativeness
        cos_sims = cosine_similarity(embs)
        r1_support = np.mean(np.sort(cos_sims[0])[-4:-1])

        # TC Formula
        # Factual truth is Dominant (l1/l2), Consistent (r1_support), and Low-Rank (1/pr)
        score = (dominance * r1_support) / (pr**0.5)

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
    engine = TC_Engine(device=device)
    scores, labels, models = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        s = engine.compute_tc(row['responses'])
        if s is not None:
            scores.append(s); labels.append(row['label']); models.append(row['model'])

    print(f"\nOverall TC AUROC: {roc_auc_score(labels, scores):.4f}")
    df_res = pd.DataFrame({'label': labels, 'score': scores, 'model': models})
    for m in df_res['model'].unique():
        print(f"Model {m} AUROC: {roc_auc_score(df_res[df_res['model']==m]['label'], df_res[df_res['model']==m]['score']):.4f}")

if __name__ == "__main__":
    run()
