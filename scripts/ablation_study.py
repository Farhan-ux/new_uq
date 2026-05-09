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

    comp = {k: norm(df[k]) for k in ['h1_max', 'closeness', 'Bayesian_GMM', 'max_contra', 'Dempster_Shafer', 'PBox_Interval', 'Fuzzy_Sets', 'stable_rank']}

    models = [
        ('llama-3.1-8b-instant', [
            ('Full: ((1-h1)*Closeness)/((1-GMM)+0.1)', lambda c: ((1-c['h1_max']) * c['closeness']) / ((1-c['Bayesian_GMM']) + 0.1)),
            ('Ablate h1', lambda c: (c['closeness']) / ((1-c['Bayesian_GMM']) + 0.1)),
            ('Ablate Closeness', lambda c: (1-c['h1_max']) / ((1-c['Bayesian_GMM']) + 0.1)),
            ('Ablate Bayesian', lambda c: (1-c['h1_max']) * c['closeness'])
        ]),
        ('qwen/qwen3-32b', [
            ('Full: ((1-Contra)*DS)/(PBox+0.1)', lambda c: ((1-c['max_contra']) * c['Dempster_Shafer']) / (c['PBox_Interval'] + 0.1)),
            ('Ablate Contra', lambda c: (c['Dempster_Shafer']) / (c['PBox_Interval'] + 0.1)),
            ('Ablate DS', lambda c: (1-c['max_contra']) / (c['PBox_Interval'] + 0.1)),
            ('Ablate PBox', lambda c: (1-c['max_contra']) * c['Dempster_Shafer'])
        ]),
        ('meta-llama/llama-4-scout-17b-16e-instruct', [
            ('Full: ((1-h1)*(1-Fuzzy))/(Rank+0.1)', lambda c: ((1-c['h1_max']) * (1-c['Fuzzy_Sets'])) / (c['stable_rank'] + 0.1)),
            ('Ablate h1', lambda c: (1-c['Fuzzy_Sets']) / (c['stable_rank'] + 0.1)),
            ('Ablate Fuzzy', lambda c: (1-c['h1_max']) / (c['stable_rank'] + 0.1)),
            ('Ablate Rank', lambda c: (1-c['h1_max']) * (1-c['Fuzzy_Sets']))
        ])
    ]

    for model_id, tasks in models:
        print(f"\n--- Ablation: {model_id} ---")
        df_m = df[df['model'] == model_id]
        indices = df_m.index
        c_m = {k: v.loc[indices] for k, v in comp.items()}
        for label, func in tasks:
            score = func(c_m)
            auc = evaluate(df_m, score)
            print(f"  {label:40s}: AUROC={auc:.4f}")

if __name__ == "__main__":
    main()
