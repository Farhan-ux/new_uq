import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import itertools

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

    cols = ['h0_max', 'h1_max', 'stable_rank', 'page_rank', 'closeness', 'lex_red', 'max_contra', 'entail_r1',
            'Fuzzy_Sets', 'Dempster_Shafer', 'Possibility', 'PBox_Interval', 'Bayesian_GMM']

    sigs_df = df[cols].apply(norm)

    models = df['model'].unique()
    results = {}

    for model in models:
        print(f"\n🚀 Searching for {model}...")
        mask = df['model'] == model
        y_m = df[mask]['label']
        sigs_m = sigs_df[mask]

        best_auc = 0
        best_form = ""

        # Optimization: prioritize likely signals
        keys = cols

        for r in [2, 3]:
            for combo in itertools.permutations(keys, r):
                for bits in itertools.product([0, 1], repeat=r):
                    parts = []
                    for i, k in enumerate(combo):
                        s = sigs_m[k]
                        if bits[i] == 1: s = 1 - s
                        parts.append(s)

                    # Product
                    score = np.prod(parts, axis=0)
                    auc = evaluate(df[mask], score)
                    if auc > best_auc:
                        best_auc = auc
                        best_form = " * ".join([f"(1-{combo[i]})" if bits[i] == 1 else combo[i] for i in range(r)])

                    # Division
                    if r == 3:
                        score = (parts[0] * parts[1]) / (parts[2] + 0.1)
                        auc = evaluate(df[mask], score)
                        if auc > best_auc:
                            best_auc = auc
                            best_form = f"({f'(1-{combo[0]})' if bits[0] == 1 else combo[0]} * {f'(1-{combo[1]})' if bits[1] == 1 else combo[1]}) / ({f'(1-{combo[2]})' if bits[2] == 1 else combo[2]} + 0.1)"

        print(f"  ✅ Best for {model}: {best_auc:.4f} | {best_form}")
        results[model] = (best_auc, best_form)

if __name__ == "__main__":
    main()
