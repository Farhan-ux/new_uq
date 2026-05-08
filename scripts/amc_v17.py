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
    b1, b2 = s1.encode(), s2.encode()
    c1, c2 = len(zlib.compress(b1)), len(zlib.compress(b2))
    c12 = len(zlib.compress(b1 + b2))
    return (c12 - min(c1, c2)) / max(c1, c2)

class MTE_Engine:
    def __init__(self, device="cpu"):
        self.device = device
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def compute_mte(self, resps):
        clean = [r for r in resps if r.strip()]
        n = len(clean)
        if n < 4: return None
        embs = self.embed_model.encode(clean)

        # 1. Axis 1: Topological Stability (H0 / (1+H1))
        pca = PCA(n_components=min(n-1, 10))
        emb_red = pca.fit_transform(embs)
        res_tda = ripser(emb_red, maxdim=1)
        h0_max = np.max(res_tda['dgms'][0][np.isfinite(res_tda['dgms'][0][:, 1])][:, 1]) if len(res_tda['dgms'][0][np.isfinite(res_tda['dgms'][0][:, 1])]) > 0 else 0
        h1 = res_tda['dgms'][1]
        h1_max = np.max(h1[:, 1] - h1[:, 0]) if len(h1) > 0 else 0
        topo_signal = h0_max / (1.0 + h1_max)

        # 2. Axis 2: Spectral Consensus (Centrality / PR)
        evs = pca.explained_variance_
        pr = (np.sum(evs)**2) / (np.sum(evs**2) + 1e-9)
        dist_mat = euclidean_distances(embs)
        sigma = np.median(dist_mat[np.triu_indices(n, k=1)]) or 0.5
        gamma = np.mean(np.exp(-(dist_mat**2)/(2*sigma**2)), axis=1)
        spec_signal = gamma[0] / (pr + 1e-9)

        # 3. Axis 3: Algorithmic Lexical Support
        # Mean NCD of r1 to others
        ncds = [get_ncd(clean[0], clean[j]) for j in range(1, n)]
        lex_signal = 1.0 / (np.mean(ncds) + 1e-9)

        # MTE: Geometric Mean of three orthogonal formulaic signals
        # log-space addition for robustness
        combined = np.log1p(topo_signal) + np.log1p(spec_signal) + np.log1p(lex_signal)
        return combined

def run():
    df_responses = pd.read_parquet('data/pilot/responses_100/responses.parquet')
    df_prompts = pd.read_parquet('data/pilot/pilot_100_prompts.parquet')
    df_responses = df_responses[df_responses['model'] != 'llama-3.3-70b-versatile'].copy()
    label_map = {'factual': 1, 'adversarial': 0}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label']], on='prompt_id')
    df = df[df['label'].isin([0, 1])].copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = MTE_Engine(device=device)
    scores, labels, models = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        s = engine.compute_mte(row['responses'])
        if s is not None:
            scores.append(s); labels.append(row['label']); models.append(row['model'])

    print(f"\nOverall MTE AUROC: {roc_auc_score(labels, scores):.4f}")
    df_res = pd.DataFrame({'label': labels, 'score': scores, 'model': models})
    for m in df_res['model'].unique():
        print(f"Model {m} AUROC: {roc_auc_score(df_res[df_res['model']==m]['label'], df_res[df_res['model']==m]['score']):.4f}")

if __name__ == "__main__":
    run()
