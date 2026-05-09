import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from zero_shot_uq import ZeroShotLlama3UQ, ZeroShotQwenUQ, ZeroShotScoutUQ
from tqdm import tqdm
import time

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece = 0
    for i in range(n_bins):
        mask = binids == i
        if np.any(mask):
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            ece += np.abs(acc - conf) * np.sum(mask)
    return ece / len(y_true)

def bootstrap_auc(y_true, y_prob, n_boot=1000):
    aucs = []
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2: continue
        aucs.append(roc_auc_score(y_true[indices], y_prob[indices]))
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

def main():
    with open('test_ids.txt', 'r') as f:
        test_ids = f.read().splitlines()

    df_resps = pd.read_parquet('data/pilot/responses_100/responses.parquet')
    df_p = pd.read_parquet('data/pilot/pilot_100_prompts.parquet')

    label_map = {'factual': 1, 'adversarial': 0}
    df_p['label'] = df_p['difficulty_type'].map(label_map)
    df = pd.merge(df_resps, df_p[['prompt_id', 'label']], on='prompt_id')
    df = df[df['prompt_id'].isin(test_ids)].copy()

    models = {
        'llama-3.1-8b-instant': ZeroShotLlama3UQ(),
        'qwen/qwen3-32b': ZeroShotQwenUQ(),
        'meta-llama/llama-4-scout-17b-16e-instruct': ZeroShotScoutUQ()
    }

    results = []

    for model_name, uq_engine in models.items():
        print(f"Validating Zero-Shot: {model_name}...")
        df_m = df[df['model'] == model_name]

        y_true, y_prob = [], []
        runtimes = []

        for _, row in tqdm(df_m.iterrows(), total=len(df_m)):
            start = time.time()
            p = uq_engine.compute(row['responses'])
            runtimes.append(time.time() - start)
            y_true.append(row['label'])
            y_prob.append(p)

        y_true, y_prob = np.array(y_true), np.array(y_prob)

        auc_mean, auc_low, auc_high = bootstrap_auc(y_true, y_prob)
        ece = compute_ece(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)

        results.append({
            'Model': model_name,
            'AUROC': f"{auc_mean:.3f} [{auc_low:.3f}, {auc_high:.3f}]",
            'ECE': f"{ece:.3f}",
            'Brier': f"{brier:.3f}",
            'Runtime': f"{np.mean(runtimes):.3f}s",
            'Range': f"[{y_prob.min():.3f}, {y_prob.max():.3f}]"
        })

    res_df = pd.DataFrame(results)
    print("\n--- ZERO-SHOT VALIDATION (30 HELD-OUT PROMPTS) ---")
    print(res_df.to_string(index=False))
    res_df.to_csv('zero_shot_results.csv', index=False)

if __name__ == "__main__":
    main()
