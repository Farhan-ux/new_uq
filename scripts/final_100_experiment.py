import pandas as pd
import numpy as np
import torch
import zlib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import AgglomerativeClustering
from ripser import ripser
import scipy.stats
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_ncd(s1, s2):
    try:
        b1, b2 = s1.encode(), s2.encode()
        c1, c2 = len(zlib.compress(b1)), len(zlib.compress(b2))
        c12 = len(zlib.compress(b1 + b2))
        return (c12 - min(c1, c2)) / max(c1, c2)
    except: return 1.0

class FinalUQEngine:
    def __init__(self, device="cpu"):
        self.device = device
        print(f"Loading models on {device}...")
        self.nli_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-small")
        self.nli_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-small").to(self.device)
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def get_nli_matrix(self, resps):
        n = len(resps)
        if n == 0: return np.zeros((0,0))
        pairs = []
        for i in range(n):
            for j in range(n):
                pairs.append((resps[i], resps[j]))

        batch_size = 128
        all_probs = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            encoded = self.nli_tokenizer([p[0] for p in batch], [p[1] for p in batch],
                                       padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.nli_model(**encoded).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)

        probs = np.vstack(all_probs)
        scores = probs[:, 0] - probs[:, 2]
        return scores.reshape(n, n)

    def compute_metrics(self, resps):
        clean = [r for r in resps if r.strip()]
        if len(clean) < 3:
            return None

        # 1. Embeddings and PCA
        embs = self.embed_model.encode(clean)
        n_comp = min(len(clean) - 1, 10)
        pca = PCA(n_components=n_comp)
        emb_reduced = pca.fit_transform(embs)
        evr = pca.explained_variance_ratio_
        id_proxy = np.sum(evr > 0.05)

        # 2. Adaptive Bandwidth
        dist_matrix = euclidean_distances(embs)
        triu_idx = np.triu_indices(len(clean), k=1)
        sigma = np.median(dist_matrix[triu_idx])
        if sigma == 0: sigma = 0.5

        # 3. TDA on reduced space
        res_tda = ripser(emb_reduced, maxdim=0)
        h0 = res_tda['dgms'][0]
        h0_finite = h0[np.isfinite(h0[:, 1])]
        h_max = np.max(h0_finite[:, 1]) if len(h0_finite) > 0 else 0

        # 4. Heat Kernel Centrality
        kernel_matrix = np.exp(-(dist_matrix**2) / (2 * sigma**2))
        gammas = np.mean(kernel_matrix, axis=1)
        gamma_r1 = gammas[0]
        gamma_median = np.median(gammas)

        # --- Novel Methods ---
        # iTME (as per image)
        itme_score = (gamma_r1 / (gamma_median + 1e-9)) * np.tanh(h_max)
        # Probabilistic mapping for itme
        p_itme = sigmoid(2.0 * itme_score - 2.0)

        # SAME (Semantic-Algorithmic Manifold Evidence)
        # Factor in intrinsic dimensionality to penalize noisy high-spread ensembles
        same_score = (h_max * (gamma_r1 / (gamma_median + 1e-9))) / (id_proxy + 1)
        p_same = sigmoid(4.0 * same_score - 2.0)

        # --- Baselines ---
        cos_sim = np.mean(cosine_similarity(embs))
        nli_mat = self.get_nli_matrix(clean)
        dist_mat = 1.0 - (nli_mat + 1.0) / 2.0
        np.fill_diagonal(dist_mat, 0)
        clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average', distance_threshold=0.5)
        clusters = clustering.fit_predict(dist_mat)
        counts = np.bincount(clusters)
        sem_entropy = scipy.stats.entropy(counts / len(clean))
        adj = (nli_mat > 0.0).astype(float)
        deg_mat = np.mean(np.sum(adj, axis=1))
        ecc = np.mean(np.max(dist_mat, axis=1))

        return {
            "iTME": p_itme,
            "SAME": p_same,
            "Semantic_Entropy": sem_entropy,
            "Semantic_Density": -cos_sim,
            "DegMat": -deg_mat,
            "Eccentricity": ecc
        }

def run_final_experiment():
    df_responses = pd.read_parquet('data/pilot/responses_100/responses.parquet')
    df_prompts = pd.read_parquet('data/pilot/pilot_100_prompts.parquet')
    exclude_model = 'llama-3.3-70b-versatile'
    df_responses = df_responses[df_responses['model'] != exclude_model].copy()
    label_map = {'factual': 1, 'adversarial': 0, 'ambiguous': -1}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label', 'difficulty_type']], on='prompt_id')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    uq = FinalUQEngine(device=device)

    all_metrics = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Final Experiment"):
        res = uq.compute_metrics(row['responses'])
        if res:
            res.update({'prompt_id': row['prompt_id'], 'model': row['model'], 'label': row['label']})
            res['difficulty_type'] = row['difficulty_type'] if 'difficulty_type' in row else row['difficulty_type_y']
            all_metrics.append(res)

    df_res = pd.DataFrame(all_metrics)
    df_res.to_parquet("final_experiment_100_full.parquet")
    df_eval = df_res[df_res['label'].isin([0, 1])].copy()

    results = []
    methods = ["iTME", "SAME", "Semantic_Entropy", "Semantic_Density", "DegMat", "Eccentricity"]
    for model in df_eval['model'].unique():
        df_m = df_eval[df_eval['model'] == model]
        for m in methods:
            auc_val = roc_auc_score(df_m['label'], df_m[m] if m in ["iTME", "SAME"] else -df_m[m])
            ece_val = compute_ece(df_m['label'].values, df_m[m].values) if m in ["iTME", "SAME"] else np.nan
            results.append({"Model": model, "Method": m, "AUROC": auc_val, "ECE": ece_val})

    # Overall
    for m in methods:
        auc_val = roc_auc_score(df_eval['label'], df_eval[m] if m in ["iTME", "SAME"] else -df_eval[m])
        ece_val = compute_ece(df_eval['label'].values, df_eval[m].values) if m in ["iTME", "SAME"] else np.nan
        results.append({"Model": "OVERALL", "Method": m, "AUROC": auc_val, "ECE": ece_val})

    summary_df = pd.DataFrame(results)
    summary_df.to_csv("final_experiment_results_100.csv", index=False)
    print("\n--- Final Consolidated Results ---")
    print(summary_df[summary_df['Model'] == 'OVERALL'])

if __name__ == "__main__":
    run_final_experiment()
