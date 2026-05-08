import pandas as pd
import numpy as np
import torch
import zlib
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from ripser import ripser
import scipy.special
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def get_ensemble_ncd(resps):
    # Normalized ensemble complexity
    individual_sizes = [len(zlib.compress(r.encode())) for r in resps]
    joint_size = len(zlib.compress(" ".join(resps).encode()))
    # Compression ratio: how much did we save by joint compression?
    # High ratio = high redundancy = consistent truth
    ratio = np.sum(individual_sizes) / (joint_size + 1e-9)
    return ratio

class AMC_V7:
    def __init__(self, device="cpu"):
        self.device = device
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def compute_metrics(self, resps):
        clean = [r for r in resps if r.strip()]
        if len(clean) < 4: return None
        embs = self.embed_model.encode(clean)

        # 1. Algorithmic Redundancy (NCD proxy)
        redundancy = get_ensemble_ncd(clean)

        # 2. Persistence Stability
        pca = PCA(n_components=min(len(clean)-1, 10))
        emb_red = pca.fit_transform(embs)
        res_tda = ripser(emb_red, maxdim=1)
        h0_max = np.max(res_tda['dgms'][0][np.isfinite(res_tda['dgms'][0][:, 1])][:, 1]) if len(res_tda['dgms'][0][np.isfinite(res_tda['dgms'][0][:, 1])]) > 0 else 0
        h1_max = np.max(res_tda['dgms'][1][:, 1] - res_tda['dgms'][1][:, 0]) if len(res_tda['dgms'][1]) > 0 else 0
        stability = h0_max / (1.0 + h1_max)

        # 3. Spectral Centrality
        dist_matrix = euclidean_distances(embs)
        sigma = np.median(dist_matrix[np.triu_indices(len(clean), k=1)])
        if sigma == 0: sigma = 0.5
        gamma = np.mean(np.exp(-(dist_matrix**2) / (2 * sigma**2)), axis=1)
        gamma_rel = gamma[0] / (np.median(gamma) + 1e-9)

        # AMC V7: Redundancy * Stability * Centrality
        score = redundancy * stability * gamma_rel

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
    amc = AMC_V7(device=device)
    scores, labels, models = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        s = amc.compute_metrics(row['responses'])
        if s is not None:
            scores.append(s); labels.append(row['label']); models.append(row['model'])

    df_res = pd.DataFrame({'label': labels, 'score': scores, 'model': models})
    print(f"\nOverall AMC V7 AUROC: {roc_auc_score(labels, scores):.4f}")
    for m in df_res['model'].unique():
        print(f"Model {m} AUROC: {roc_auc_score(df_res[df_res['model']==m]['label'], df_res[df_res['model']==m]['score']):.4f}")

if __name__ == "__main__":
    run()
