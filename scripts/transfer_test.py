import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def evaluate(df_m, score):
    y = df_m['label']
    try:
        auc = roc_auc_score(y, score)
        if auc < 0.5: auc = 1 - auc
        return auc
    except:
        return 0.5

def main():
    df = pd.read_parquet('arch_features_100.parquet')
    def norm(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)

    # Pre-calculate components for the selected formulas
    # Formula 1 (Llama 3.1): ((1-h1_max) * closeness) / ((1-Bayesian_GMM) + 0.1)
    # Formula 2 (Qwen): ((1-max_contra) * Dempster_Shafer) / (PBox_Interval + 0.1)
    # Formula 3 (Scout): ((1-h1_max) * (1-Fuzzy_Sets)) / (stable_rank + 0.1)

    comp = {k: norm(df[k]) for k in ['h1_max', 'closeness', 'Bayesian_GMM', 'max_contra', 'Dempster_Shafer', 'PBox_Interval', 'Fuzzy_Sets', 'stable_rank']}

    f1 = ((1 - comp['h1_max']) * comp['closeness']) / ((1 - comp['Bayesian_GMM']) + 0.1)
    f2 = ((1 - comp['max_contra']) * comp['Dempster_Shafer']) / (comp['PBox_Interval'] + 0.1)
    f3 = ((1 - comp['h1_max']) * (1 - comp['Fuzzy_Sets'])) / (comp['stable_rank'] + 0.1)

    models = [
        'llama-3.1-8b-instant',
        'qwen/qwen3-32b',
        'meta-llama/llama-4-scout-17b-16e-instruct'
    ]

    methods = [('Llama-3.1-Specific', f1), ('Qwen-Specific', f2), ('Scout-Specific', f3)]

    results = []
    for m_name, score in methods:
        row = {'Method': m_name}
        for model in models:
            mask = df['model'] == model
            auc = evaluate(df[mask], score[mask])
            row[model] = auc
        results.append(row)

    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    main()
