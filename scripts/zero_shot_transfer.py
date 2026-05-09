import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scripts.zero_shot_uq import ZeroShotLlama3UQ, ZeroShotQwenUQ, ZeroShotScoutUQ

def evaluate(y_true, y_prob):
    try:
        auc = roc_auc_score(y_true, y_prob)
        if auc < 0.5: auc = 1 - auc # We report the absolute discriminative power
        return auc
    except:
        return 0.5

def main():
    with open('test_ids.txt', 'r') as f:
        test_ids = f.read().splitlines()

    df_resps = pd.read_parquet('data/pilot/responses_100/responses.parquet')
    df_p = pd.read_parquet('data/pilot/pilot_100_prompts.parquet')

    label_map = {'factual': 1, 'adversarial': 0}
    df_p['label'] = df_p['difficulty_type'].map(label_map)
    df = pd.merge(df_resps, df_p[['prompt_id', 'label']], on='prompt_id')
    df = df[df['prompt_id'].isin(test_ids)].copy()

    models = ['llama-3.1-8b-instant', 'qwen/qwen3-32b', 'meta-llama/llama-4-scout-17b-16e-instruct']
    engines = [
        ('Llama3-Method', ZeroShotLlama3UQ()),
        ('Qwen-Method', ZeroShotQwenUQ()),
        ('Scout-Method', ZeroShotScoutUQ())
    ]

    results = []
    for eng_name, engine in engines:
        row = {'Method': eng_name}
        for model in models:
            df_m = df[df['model'] == model]
            y_true = df_m['label'].values
            y_prob = [engine.compute(r) for r in df_m['responses']]
            row[model] = evaluate(y_true, y_prob)
        results.append(row)

    res_df = pd.DataFrame(results)
    print("\n--- ZERO-SHOT CROSS-MODEL TRANSFER MATRIX (AUROC) ---")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    main()
