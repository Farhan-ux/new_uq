import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from ripser import ripser
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import time
import os

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    bin_total = np.bincount(binids, minlength=n_bins)
    nonzero = bin_total > 0
    bin_probs = np.bincount(binids, weights=y_prob, minlength=n_bins)[nonzero] / bin_total[nonzero]
    bin_acc = np.bincount(binids, weights=y_true, minlength=n_bins)[nonzero] / bin_total[nonzero]
    return np.sum(np.abs(bin_probs - bin_acc) * bin_total[nonzero]) / len(y_true)

class iTME:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    def compute_score(self, resps):
        clean = [r for r in resps if r.strip()]
        if len(clean) < 3: return 0.5

        embeddings = self.model.encode(clean)

        # Fix 1: PCA Pre-processing
        # With 10 samples, max components is 9
        n_comp = min(len(clean) - 1, 10)
        pca = PCA(n_components=0.95 if len(clean) > 5 else n_comp)
        try:
            emb_reduced = pca.fit_transform(embeddings)
        except:
            emb_reduced = embeddings # fallback

        # Fix 2: Adaptive Bandwidth sigma
        dist_matrix = euclidean_distances(embeddings)
        triu_idx = np.triu_indices(len(clean), k=1)
        sigma = np.median(dist_matrix[triu_idx])
        if sigma == 0: sigma = 0.5

        # TDA on reduced space
        res_tda = ripser(emb_reduced, maxdim=0)
        h0 = res_tda['dgms'][0]
        h_max = np.max(h0[np.isfinite(h0[:, 1]), 1]) if len(h0) > 0 else 0

        # Gamma calculation (heat kernel)
        # Using original embeddings or reduced? Image says "10 embeddings" for sigma.
        # Let's use original for gamma as it's more stable for density.
        kernel_matrix = np.exp(-(dist_matrix**2) / (2 * sigma**2))
        gammas = np.mean(kernel_matrix, axis=1)
        gamma_r1 = gammas[0]
        gamma_median = np.median(gammas)

        # Fix 3: Contrastive Scoring
        # S = (gamma_r1 / gamma_median) * tanh(h_max)
        score = (gamma_r1 / (gamma_median + 1e-9)) * np.tanh(h_max)

        # Map to [0, 1] using a simple sigmoid for probability interpretation
        # Since score can be > 1, we might need a scaling factor.
        # But image just says "New Score". I'll use sigmoid(score - 1) or similar.
        # Let's try to keep it as raw score first to see AUROC, then calibrate.
        return score

def run_itme():
    df_responses = pd.read_parquet('data/pilot/responses_100/responses.parquet')
    df_prompts = pd.read_parquet('data/pilot/pilot_100_prompts.parquet')

    exclude_model = 'llama-3.3-70b-versatile'
    df_responses = df_responses[df_responses['model'] != exclude_model].copy()

    label_map = {'factual': 1, 'adversarial': 0, 'ambiguous': -1}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label']], on='prompt_id')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    itme_engine = iTME(device=device)

    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="iTME"):
        s = itme_engine.compute_score(row['responses'])
        results.append({'idx': idx, 'itme_score': s, 'label': row['label'], 'model': row['model']})

    res_df = pd.DataFrame(results)
    eval_df = res_df[res_df['label'].isin([0, 1])]

    summary = []
    for model in eval_df['model'].unique():
        m_df = eval_df[eval_df['model'] == model]
        auc = roc_auc_score(m_df['label'], m_df['itme_score'])
        summary.append({'Model': model, 'iTME_AUROC': auc})

    overall_auc = roc_auc_score(eval_df['label'], eval_df['itme_score'])
    summary.append({'Model': 'OVERALL', 'iTME_AUROC': overall_auc})

    print("\n--- iTME Results ---")
    print(pd.DataFrame(summary))

if __name__ == "__main__":
    run_itme()
