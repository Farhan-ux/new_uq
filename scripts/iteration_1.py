import pandas as pd
import numpy as np
import torch
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import euclidean_distances
from ripser import ripser
from sklearn.metrics import roc_auc_score

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

def run_iteration_1():
    print("[ITERATION 1]")
    print("Hypothesis: Factual truth forms a persistent semantic manifold; TME uses max H0 persistence and heat-kernel centrality to score r1.")

    # 1. Load Data
    df_prompts = pd.read_parquet('data/pilot/pilot_prompts_20.parquet')
    df_responses = pd.read_parquet('data/pilot/responses/pilot_responses_groq.parquet')

    label_map = {'factual': 1, 'adversarial': 0}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label', 'difficulty_type']], on='prompt_id')
    df_eval = df[df['difficulty_type'].isin(['factual', 'adversarial'])].copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    results = []
    start_time = time.time()

    for idx, row in df_eval.iterrows():
        resps = row['responses']
        clean = [r for r in resps if r.strip()]
        if len(clean) < 2:
            results.append(0.5)
            continue

        embeddings = model.encode(clean)
        e1 = embeddings[0:1]

        # 1. Topological Signal: H0 Persistence (Manifold Breadth)
        # Higher persistence -> more valid "jitters" of the truth -> Factual
        res_tda = ripser(embeddings, maxdim=0)
        h0 = res_tda['dgms'][0]
        h0_finite = h0[np.isfinite(h0[:, 1])]
        max_h0 = np.max(h0_finite[:, 1]) if len(h0_finite) > 0 else 0

        # 2. Centrality Signal: Heat Kernel (r1 Evidence)
        # How well r1 is supported by the rest of the manifold
        dists = euclidean_distances(e1, embeddings)[0]
        # sigma = 0.5 is a standard heuristic for normalized embeddings
        kernel_vals = np.exp(- (dists**2) / (2 * 0.5**2))
        centrality = np.mean(kernel_vals)

        # 3. Principled Mapping
        # We transform the features to a log-odds scale
        # logit = log(persistence) + log(centrality) - bias
        # We use log-transform to handle the different scales of topological persistence
        logit = 2.5 * np.log(max_h0 + 1e-6) + 1.5 * np.log(centrality + 1e-6) + 3.0
        prob = sigmoid(logit)

        results.append(prob)

    avg_runtime = (time.time() - start_time) / len(df_eval)
    y_true = df_eval['label'].values
    y_prob = np.array(results)

    auroc = roc_auc_score(y_true, y_prob)
    ece = compute_ece(y_true, y_prob)

    print(f"Mathematical/Algorithmic Core: P = sigmoid(2.5*ln(H_max) + 1.5*ln(Centrality) + 3.0)")
    print(f"Pilot Results:")
    print(f"• AUROC: {auroc:.3f}")
    print(f"• ECE: {ece:.3f}")
    print(f"• Probability Range: [{y_prob.min():.3f}, {y_prob.max():.3f}]")
    print(f"• Runtime/prompt: {avg_runtime:.3f}s")

    decision = "CONTINUE" if auroc >= 0.60 and ece <= 0.20 else "REFINE"
    print(f"Decision: {decision}")

    if decision == "CONTINUE":
        # Save results for final package
        df_eval['p_factual'] = y_prob
        df_eval[['prompt_id', 'p_factual']].to_parquet('iteration_1_results.parquet')

if __name__ == "__main__":
    run_iteration_1()
