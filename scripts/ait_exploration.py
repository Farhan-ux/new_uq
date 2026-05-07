import pandas as pd
import numpy as np
import zlib
import bz2
import lzma
import itertools
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import LeaveOneOut

def get_compressed_len(text, method='zlib'):
    b = text.encode()
    if method == 'zlib': return len(zlib.compress(b))
    if method == 'bz2': return len(bz2.compress(b))
    if method == 'lzma': return len(lzma.compress(b))
    return len(b)

def main():
    df_prompts = pd.read_parquet('data/pilot/pilot_prompts_20.parquet')
    df_responses = pd.read_parquet('data/pilot/responses/pilot_responses_groq.parquet')

    label_map = {'factual': 1, 'adversarial': 0}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label', 'difficulty_type']], on='prompt_id')
    df_eval = df[df['difficulty_type'].isin(['factual', 'adversarial'])].copy()

    results = []

    methods = ['zlib', 'bz2', 'lzma']

    for m in methods:
        ait_scores = []
        for resps in df_eval['responses']:
            clean = [r for r in resps if r.strip()]
            if not clean:
                ait_scores.append({'ratio': 0, 'redundancy': 0, 'joint_complexity': 0})
                continue

            individual_lens = [get_compressed_len(r, m) for r in clean]
            total_individual = sum(individual_lens)
            joint_text = " ".join(clean)
            joint_len = get_compressed_len(joint_text, m)

            # Metric 1: Compression Ratio (Lower = more redundant/consistent)
            ratio = joint_len / total_individual if total_individual > 0 else 1.0

            # Metric 2: Redundancy Gain
            redundancy = (total_individual - joint_len) / total_individual if total_individual > 0 else 0.0

            # Metric 3: Normalized Joint Complexity
            # NCD approximation for set: (C(S) - min(C(ri))) / max(C(ri)) - not quite right for sets
            # Let's use Normalized Compression Gain

            ait_scores.append({
                f'{m}_ratio': ratio,
                f'{m}_redundancy': redundancy,
                f'{m}_joint': joint_len
            })

        scores_df = pd.DataFrame(ait_scores)
        for col in scores_df.columns:
            X = scores_df[col].values
            y = df_eval['label'].values

            # Test orig and inv
            auroc_orig = roc_auc_score(y, -X)
            auroc_inv = roc_auc_score(y, X)

            results.append({'method': f"{col}_orig", 'auroc': auroc_orig})
            results.append({'method': f"{col}_inv", 'auroc': auroc_inv})

    res_df = pd.DataFrame(results).sort_values('auroc', ascending=False)
    print(res_df.head(10))

if __name__ == "__main__":
    main()
