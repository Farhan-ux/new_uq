import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
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

class UnifiedUQ:
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

    def compute_all_metrics(self, resps):
        clean = [r for r in resps if r.strip()]
        if len(clean) < 2:
            return None

        embs = self.embed_model.encode(clean)
        res_tda = ripser(embs, maxdim=0)
        h0 = res_tda['dgms'][0]
        h0_finite = h0[np.isfinite(h0[:, 1])]
        h_max = np.max(h0_finite[:, 1]) if len(h0_finite) > 0 else 0

        dists_euc = euclidean_distances(embs[0:1], embs)[0]
        gamma = np.mean(np.exp(-(dists_euc**2) / (2 * 0.5**2)))
        tme_logit = 2.5 * np.log(h_max + 1e-6) + 1.5 * np.log(gamma + 1e-6) + 3.0
        p_tme = sigmoid(tme_logit)

        cos_sim = cosine_similarity(embs)
        sem_density = np.mean(cos_sim)

        nli_mat = self.get_nli_matrix(clean)
        dist_mat = 1.0 - (nli_mat + 1.0) / 2.0
        np.fill_diagonal(dist_mat, 0)
        clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average', distance_threshold=0.5)
        clusters = clustering.fit_predict(dist_mat)
        counts = np.bincount(clusters)
        probs = counts / len(clean)
        sem_entropy = scipy.stats.entropy(probs)

        adj = (nli_mat > 0.0).astype(float)
        deg = np.sum(adj, axis=1)
        deg_mat = np.mean(deg)
        ecc = np.mean(np.max(dist_mat, axis=1))

        def jaccard(s1, s2):
            w1, w2 = set(s1.lower().split()), set(s2.lower().split())
            return len(w1 & w2) / len(w1 | w2) if (w1 | w2) else 0
        lex_sims = [jaccard(clean[i], clean[j]) for i in range(len(clean)) for j in range(i+1, len(clean))]
        lex_sim = np.mean(lex_sims)

        return {
            "TME": p_tme,
            "Semantic_Entropy": sem_entropy,
            "Lexical_Similarity": -lex_sim,
            "Semantic_Density": -sem_density,
            "DegMat": -deg_mat,
            "Eccentricity": ecc
        }

def run_experiment():
    df_responses = pd.read_parquet('data/pilot/responses_100/responses.parquet')
    df_prompts = pd.read_parquet('data/pilot/pilot_100_prompts.parquet')

    exclude_model = 'llama-3.3-70b-versatile'
    df_responses = df_responses[df_responses['model'] != exclude_model].copy()

    label_map = {'factual': 1, 'adversarial': 0, 'ambiguous': -1}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)

    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label', 'difficulty_type']], on='prompt_id')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    uq = UnifiedUQ(device=device)

    all_metrics = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing Metrics"):
        res = uq.compute_all_metrics(row['responses'])
        if res:
            res['prompt_id'] = row['prompt_id']
            res['model'] = row['model']
            res['label'] = row['label']
            diff_type = row['difficulty_type'] if 'difficulty_type' in row else row['difficulty_type_y']
            res['difficulty_type'] = diff_type
            all_metrics.append(res)

    df_res = pd.DataFrame(all_metrics)
    df_res.to_parquet("all_metrics_100.parquet")

    df_eval = df_res[df_res['label'].isin([0, 1])].copy()

    results_summary = []
    methods = ["TME", "Semantic_Entropy", "Lexical_Similarity", "Semantic_Density", "DegMat", "Eccentricity"]

    for model in df_eval['model'].unique():
        df_m = df_eval[df_eval['model'] == model]
        for method in methods:
            y_true = df_m['label'].values
            y_score = df_m[method].values
            auc_score = roc_auc_score(y_true, -y_score if method != "TME" else y_score)
            ece_val = compute_ece(y_true, y_score) if method == "TME" else np.nan
            results_summary.append({"Model": model, "Method": method, "AUROC": auc_score, "ECE": ece_val})

    for method in methods:
        y_true = df_eval['label'].values
        y_score = df_eval[method].values
        auc_score = roc_auc_score(y_true, -y_score if method != "TME" else y_score)
        ece_val = compute_ece(y_true, y_score) if method == "TME" else np.nan
        results_summary.append({"Model": "OVERALL", "Method": method, "AUROC": auc_score, "ECE": ece_val})

    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv("experiment_results_100.csv", index=False)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for model in df_eval['model'].unique():
        df_m = df_eval[df_eval['model'] == model]
        fpr, tpr, _ = roc_curve(df_m['label'], df_m['TME'])
        plt.plot(fpr, tpr, label=f'{model} (AUC={auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('TME ROC Curve by Model')
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.kdeplot(data=df_res[df_res['label'] != -1], x='TME', hue='difficulty_type', fill=True)
    plt.title('TME Probability Distribution (100 Prompts)')
    plt.savefig('tme_100_results.png')

    print("\nResults saved to experiment_results_100.csv and tme_100_results.png")

if __name__ == "__main__":
    run_experiment()
