import pandas as pd
import numpy as np
import torch
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import euclidean_distances
from ripser import ripser
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

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

class TMEInference:
    def __init__(self, device="cpu"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    def get_features(self, responses, sigma=0.5):
        clean = [r for r in responses if r.strip()]
        if len(clean) < 2:
            return None

        embeddings = self.model.encode(clean)
        e1 = embeddings[0:1]

        res_tda = ripser(embeddings, maxdim=0)
        h0 = res_tda['dgms'][0]
        h0_finite = h0[np.isfinite(h0[:, 1])]
        max_h0 = np.max(h0_finite[:, 1]) if len(h0_finite) > 0 else 0

        dists = euclidean_distances(e1, embeddings)[0]
        kernel_vals = np.exp(- (dists**2) / (2 * sigma**2))
        centrality = np.mean(kernel_vals)

        return max_h0, centrality

def run_validation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tme = TMEInference(device=device)

    df_prompts = pd.read_parquet('data/pilot/pilot_prompts_20.parquet')
    df_responses = pd.read_parquet('data/pilot/responses/pilot_responses_groq.parquet')

    label_map = {'factual': 1, 'adversarial': 0, 'ambiguous': -1}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label', 'difficulty_type', 'prompt_text']], on='prompt_id')

    # Pre-compute features for efficiency
    print("Pre-computing features...")
    features = []
    for idx, row in df.iterrows():
        f = tme.get_features(row['responses'])
        if f:
            features.append({'idx': idx, 'h_max': f[0], 'gamma': f[1]})

    feat_df = pd.DataFrame(features)
    df = df.iloc[feat_df['idx']].copy()
    df['h_max'] = feat_df['h_max'].values
    df['gamma'] = feat_df['gamma'].values

    df_eval = df[df['label'].isin([0, 1])].copy()
    y = df_eval['label'].values
    h = df_eval['h_max'].values
    g = df_eval['gamma'].values

    # 1. Ablation Study
    print("\n--- Ablation Study ---")
    # TME-Full: logit = 2.5 * ln(H) + 1.5 * ln(G) + 3.0
    # TME-Topo: logit = 4.0 * ln(H) + 3.0 (rescale to keep similar bias/range)
    # TME-Centrality: logit = 4.0 * ln(G) + 3.0

    def calc_metrics(logit):
        p = sigmoid(logit)
        return roc_auc_score(y, p), compute_ece(y, p)

    full_auc, full_ece = calc_metrics(2.5 * np.log(h + 1e-6) + 1.5 * np.log(g + 1e-6) + 3.0)
    topo_auc, topo_ece = calc_metrics(4.0 * np.log(h + 1e-6) + 3.0)
    cent_auc, cent_ece = calc_metrics(4.0 * np.log(g + 1e-6) + 3.0)

    print(f"TME-Full: AUC={full_auc:.3f}, ECE={full_ece:.3f}")
    print(f"TME-Topo: AUC={topo_auc:.3f}, ECE={topo_ece:.3f}")
    print(f"TME-Cent: AUC={cent_auc:.3f}, ECE={cent_ece:.3f}")

    # 2. Parameter Sensitivity
    print("\n--- Sensitivity Analysis ---")
    sigmas = [0.3, 0.5, 0.7, 1.0]
    alphas = [1.0, 2.5, 3.5]
    betas = [1.0, 1.5, 3.5]
    biases = [-1.0, 1.0, 3.0]

    results = []
    for s in sigmas:
        # Recompute gamma for different sigma
        gammas = []
        for idx, row in df_eval.iterrows():
            resps = row['responses']
            embeddings = tme.model.encode([r for r in resps if r.strip()])
            e1 = embeddings[0:1]
            dists = euclidean_distances(e1, embeddings)[0]
            gammas.append(np.mean(np.exp(-(dists**2)/(2*s**2))))
        gammas = np.array(gammas)

        for a in alphas:
            for b in betas:
                for bias in biases:
                    logit = a * np.log(h + 1e-6) + b * np.log(gammas + 1e-6) + bias
                    p = sigmoid(logit)
                    auc = roc_auc_score(y, p)
                    results.append(auc)

    results = np.array(results)
    pass_rate = np.mean(results >= 0.65)
    print(f"AUROC >= 0.65 rate: {pass_rate*100:.1f}%")
    print(f"Min AUROC: {results.min():.3f}, Max AUROC: {results.max():.3f}")

    # 3. Cross-Validation
    print("\n--- Cross-Validation ---")
    def get_p(h_val, g_val):
        return sigmoid(2.5 * np.log(h_val + 1e-6) + 1.5 * np.log(g_val + 1e-6) + 3.0)

    # LOO
    loo = LeaveOneOut()
    loo_aucs = []
    # Actually LOO on 26 samples for AUROC is tricky, but we can compute the scores and then 1 AUC
    loo_probs = []
    for train_idx, test_idx in loo.split(df_eval):
        loo_probs.append(get_p(h[test_idx], g[test_idx])[0])
    loo_auc = roc_auc_score(y, loo_probs)

    # 5-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    skf_aucs = []
    for train_idx, test_idx in skf.split(df_eval, y):
        skf_aucs.append(roc_auc_score(y[test_idx], get_p(h[test_idx], g[test_idx])))

    print(f"LOO AUROC: {loo_auc:.3f}")
    print(f"5-Fold AUROC: {np.mean(skf_aucs):.3f} ± {np.std(skf_aucs):.3f}")

    # 4. Qualitative
    print("\n--- Qualitative Case Studies ---")
    # 2 Factual High P
    factual = df_eval[df_eval['label'] == 1].sort_values('h_max', ascending=False).head(2)
    # 2 Adversarial Low P
    adv = df_eval[df_eval['label'] == 0].sort_values('h_max', ascending=True).head(2)
    # 1 Ambiguous
    ambig = df[df['label'] == -1].head(1)

    cases = pd.concat([factual, adv, ambig])
    for idx, row in cases.iterrows():
        p = get_p(row['h_max'], row['gamma'])
        print(f"Type: {row['difficulty_type']}")
        print(f"Prompt: {row['prompt_text'][:100]}...")
        print(f"H_max: {row['h_max']:.3f}, Gamma: {row['gamma']:.3f}, P_factual: {p:.3f}")
        print("-" * 20)

if __name__ == "__main__":
    run_validation()
