import pandas as pd
import numpy as np
import torch
import zlib
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from ripser import ripser
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import scipy.special

def get_lexical_redundancy(resps):
    ind_sizes = [len(zlib.compress(r.encode())) for r in resps]
    joint_size = len(zlib.compress(" ".join(resps).encode()))
    return np.sum(ind_sizes) / (joint_size + 1e-9)

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    bin_total = np.bincount(binids, minlength=n_bins)
    nonzero = bin_total > 0
    bin_probs = np.bincount(binids, weights=y_prob, minlength=n_bins)[nonzero] / bin_total[nonzero]
    bin_acc = np.bincount(binids, weights=y_true, minlength=n_bins)[nonzero] / bin_total[nonzero]
    return np.sum(np.abs(bin_probs - bin_acc) * bin_total[nonzero]) / len(y_true)

class AMC_Final:
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
        res_tda = ripser(emb_red, maxdim=1)
        h0_max = np.max(res_tda['dgms'][0][np.isfinite(res_tda['dgms'][0][:, 1])][:, 1]) if len(res_tda['dgms'][0][np.isfinite(res_tda['dgms'][0][:, 1])]) > 0 else 0
        h1_max = np.max(res_tda['dgms'][1][:, 1] - res_tda['dgms'][1][:, 0]) if len(res_tda['dgms'][1]) > 0 else 0
        evs = pca.explained_variance_
        stable_rank = np.sum(evs) / (evs[0] + 1e-9)
        cos_sims = cosine_similarity(embs)
        medoid_idx = np.argmax(np.sum(cos_sims, axis=1))
        support = (cos_sims[0, medoid_idx] + np.mean(np.sort(cos_sims[0])[-4:-1])) / 2.0
        redundancy = get_lexical_redundancy(clean)
        raw_score = (h0_max / (1.0 + h1_max)) * support * redundancy / stable_rank
        p_amc = float(scipy.special.expit(12.0 * raw_score - 3.0))
        return p_amc

def run():
    df_responses = pd.read_parquet('data/pilot/responses_100/responses.parquet')
    df_prompts = pd.read_parquet('data/pilot/pilot_100_prompts.parquet')
    df_responses = df_responses[df_responses['model'] != 'llama-3.3-70b-versatile'].copy()
    label_map = {'factual': 1, 'adversarial': 0}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label']], on='prompt_id')
    df = df[df['label'].isin([0, 1])].copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amc = AMC_Final(device=device)
    scores, labels, models = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        s = amc.compute_score(row['responses'])
        if s is not None:
            scores.append(s); labels.append(row['label']); models.append(row['model'])

    auc = roc_auc_score(labels, scores)
    ece = compute_ece(np.array(labels), np.array(scores))
    print(f"\n--- AMC FINAL RESULTS ---")
    print(f"Overall AUROC: {auc:.4f}")
    print(f"Overall ECE: {ece:.4f}")
    df_res = pd.DataFrame({'label': labels, 'score': scores, 'model': models})
    for m in df_res['model'].unique():
        m_auc = roc_auc_score(df_res[df_res['model']==m]['label'], df_res[df_res['model']==m]['score'])
        print(f"Model {m} AUROC: {m_auc:.4f}")
    df_res.to_csv("amc_final_results.csv", index=False)

if __name__ == "__main__":
    run()
