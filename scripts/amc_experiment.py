import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from ripser import ripser
import scipy.special
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    bin_total = np.bincount(binids, minlength=n_bins)
    nonzero = bin_total > 0
    bin_probs = np.bincount(binids, weights=y_prob, minlength=n_bins)[nonzero] / bin_total[nonzero]
    bin_acc = np.bincount(binids, weights=y_true, minlength=n_bins)[nonzero] / bin_total[nonzero]
    return np.sum(np.abs(bin_probs - bin_acc) * bin_total[nonzero]) / len(y_true)

class AMC_Engine:
    def __init__(self, device="cpu"):
        self.device = device
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def compute_amc(self, resps):
        clean = [r for r in resps if r.strip()]
        if len(clean) < 4: return None
        
        # 1. Embeddings
        embs = self.embed_model.encode(clean)
        
        # 2. PCA & Participation Ratio (Smooth ID)
        n_comp = min(len(clean) - 1, 10)
        pca = PCA(n_components=n_comp)
        emb_reduced = pca.fit_transform(embs)
        evs = pca.explained_variance_
        pr = (np.sum(evs)**2) / (np.sum(evs**2) + 1e-9)
        
        # 3. Persistent Homology (H0 and H1)
        # H1 persistence captures semantic loops/confusion
        res_tda = ripser(emb_reduced, maxdim=1)
        
        h0 = res_tda['dgms'][0]
        h0_finite = h0[np.isfinite(h0[:, 1])]
        h0_max = np.max(h0_finite[:, 1]) if len(h0_finite) > 0 else 0
        
        h1 = res_tda['dgms'][1]
        h1_max = np.max(h1[:, 1] - h1[:, 0]) if len(h1) > 0 else 0
        
        # 4. Centrality
        dist_matrix = euclidean_distances(embs)
        sigma = np.median(dist_matrix[np.triu_indices(len(clean), k=1)])
        if sigma == 0: sigma = 0.5
        kernel_matrix = np.exp(-(dist_matrix**2) / (2 * sigma**2))
        gammas = np.mean(kernel_matrix, axis=1)
        gamma_rel = gammas[0] / (np.median(gammas) + 1e-9)
        
        # 5. AMC Score
        # Reward H0 persistence and primary support
        # Penalize intrinsic dimensionality and semantic cycles
        amc_score = (h0_max * gamma_rel) / (pr * (1.0 + h1_max))
        
        # 6. Probabilistic Mapping (Sigmoid)
        # Shifted to accommodate AMC scale
        p_amc = float(scipy.special.expit(8.0 * amc_score - 4.0))
        
        return p_amc

def run():
    df_responses = pd.read_parquet('data/pilot/responses_100/responses.parquet')
    df_prompts = pd.read_parquet('data/pilot/pilot_100_prompts.parquet')
    exclude_model = 'llama-3.3-70b-versatile'
    df_responses = df_responses[df_responses['model'] != exclude_model].copy()
    label_map = {'factual': 1, 'adversarial': 0}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label']], on='prompt_id')
    df = df[df['label'].isin([0, 1])].copy()
    
    amc = AMC_Engine(device="cuda" if torch.cuda.is_available() else "cpu")
    
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating AMC"):
        p = amc.compute_amc(row['responses'])
        if p is not None:
            results.append({'prompt_id': row['prompt_id'], 'model': row['model'], 'label': row['label'], 'AMC': p})
            
    df_res = pd.DataFrame(results)
    
    # Eval
    auc_val = roc_auc_score(df_res['label'], df_res['AMC'])
    ece_val = compute_ece(df_res['label'].values, df_res['AMC'].values)
    
    print(f"\n--- AMC Performance ---")
    print(f"Overall AUROC: {auc_val:.4f}")
    print(f"Overall ECE: {ece_val:.4f}")
    
    # Model breakdown
    for model in df_res['model'].unique():
        df_m = df_res[df_res['model'] == model]
        m_auc = roc_auc_score(df_m['label'], df_m['AMC'])
        print(f"Model {model} AUROC: {m_auc:.4f}")

if __name__ == "__main__":
    run()
