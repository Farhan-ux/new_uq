import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scripts.zero_shot_uq import BaseZeroShotUQ
from tqdm import tqdm

def main():
    with open('test_ids.txt', 'r') as f:
        test_ids = f.read().splitlines()

    df_resps = pd.read_parquet('data/pilot/responses_100/responses.parquet')
    df_p = pd.read_parquet('data/pilot/pilot_100_prompts.parquet')
    df_p['label'] = df_p['difficulty_type'].map({'factual': 1, 'adversarial': 0})
    df = pd.merge(df_resps, df_p[['prompt_id', 'label']], on='prompt_id')
    df = df[df['prompt_id'].isin(test_ids)].copy()

    base = BaseZeroShotUQ()
    models = df['model'].unique()

    signals = ['h1_max', 'h0_max', 'closeness', 'gmm_prob', 'max_contra', 'belief', 'pbox_width', 'fuzzy_ent', 'stable_rank', 'ev1_ratio']

    results = []
    for model_name in models:
        df_m = df[df['model'] == model_name]
        y_true = df_m['label'].values

        feats_list = []
        for r in tqdm(df_m['responses'], desc=model_name):
            feats_list.append(base.get_features(r))

        for s in signals:
            vals = []
            for f in feats_list:
                if f and s in f:
                    vals.append(f[s])
                else:
                    vals.append(0.5)

            vals = np.array(vals)
            try:
                auc = roc_auc_score(y_true, vals)
                if auc < 0.5: auc = 1 - auc
                results.append({'Model': model_name, 'Signal': s, 'AUC': auc})
            except:
                pass

    res_df = pd.DataFrame(results).pivot(index='Signal', columns='Model', values='AUC')
    print(res_df)

if __name__ == "__main__":
    main()
