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

def run_iteration_2():
    print("[ITERATION 2]")
    print("Hypothesis: Factual truth manifolds exhibit higher Persistence Entropy (diversity of structure) than singular hallucinations.")

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
        if len(clean) < 3:
            results.append(0.5)
            continue

        embeddings = model.encode(clean)
        e1 = embeddings[0:1]

        # 1. Topological Signal: Persistence Entropy
        res_tda = ripser(embeddings, maxdim=0)
        h0 = res_tda['dgms'][0]
        h0_finite = h0[np.isfinite(h0[:, 1])]
        if len(h0_finite) > 1:
            pers = h0_finite[:, 1]
            # Filter out very small noise
            pers = pers[pers > 0.01]
            if len(pers) > 0:
                p = pers / pers.sum()
                ent = -np.sum(p * np.log(p + 1e-9))
            else:
                ent = 0
        else:
            ent = 0

        # 2. Centrality Signal: Neighborhood Density
        # Using a fixed radius R=0.4 (approx semantic cluster size)
        dists = euclidean_distances(e1, embeddings)[0]
        density = np.sum(dists < 0.4) / len(clean)

        # 3. Principled Mapping
        # logit = scale * (entropy + density) + bias
        logit = 4.0 * (ent + density) - 6.0
        prob = sigmoid(logit)

        results.append(prob)

    avg_runtime = (time.time() - start_time) / len(df_eval)
    y_true = df_eval['label'].values
    y_prob = np.array(results)

    auroc = roc_auc_score(y_true, y_prob)
    ece = compute_ece(y_true, y_prob)

    print(f"Mathematical/Algorithmic Core: P = sigmoid(4.0 * (H_entropy + Density_0.4) - 6.0)")
    print(f"Pilot Results:")
    print(f"• AUROC: {auroc:.3f}")
    print(f"• ECE: {ece:.3f}")
    print(f"• Probability Range: [{y_prob.min():.3f}, {y_prob.max():.3f}]")
    print(f"• Runtime/prompt: {avg_runtime:.3f}s")

    decision = "REFINE" if auroc < 0.72 else "CONTINUE"
    print(f"Decision: {decision}")

if __name__ == "__main__":
    run_iteration_2()
