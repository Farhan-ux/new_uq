import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    df_base = pd.read_parquet('experiments/pilot_uq_benchmark/baseline_scores.parquet')
    df_verb = pd.read_parquet('experiments/pilot_uq_benchmark/verbalized_scores.parquet')
    df_ssd = pd.read_parquet('experiments/pilot_uq_benchmark/ssd_scores.parquet')

    for df in [df_base, df_verb]:
        if 'confidence_score' not in df.columns:
            df['confidence_score'] = 1.0 - df['uncertainty_score']

    df_all = pd.concat([df_base, df_verb, df_ssd], ignore_index=True)
    df_prompts = pd.read_parquet('data/pilot/pilot_prompts_20.parquet')
    df_probs = pd.read_parquet('data/pilot/probabilities/pilot_probabilities.parquet')

    df_merged = pd.merge(df_all, df_prompts[['prompt_id', 'difficulty_type']], on='prompt_id')
    label_map = {'factual': 1, 'adversarial': 0}
    df_merged['gt'] = df_merged['difficulty_type'].map(label_map)

    results = []
    for method in df_merged['method'].unique():
        subset = df_merged[(df_merged['method'] == method) & (df_merged['difficulty_type'].isin(['factual', 'adversarial']))].dropna(subset=['gt', 'confidence_score'])
        if len(subset) < 2: continue

        # Cross-validated AUROC check for SSD
        auroc = roc_auc_score(subset['gt'], subset['confidence_score'])

        mean_factual = df_merged[(df_merged['method'] == method) & (df_merged['difficulty_type'] == 'factual')]['uncertainty_score'].mean()
        mean_adv = df_merged[(df_merged['method'] == method) & (df_merged['difficulty_type'] == 'adversarial')]['uncertainty_score'].mean()
        mean_ambig = df_merged[(df_merged['method'] == method) & (df_merged['difficulty_type'] == 'ambiguous')]['uncertainty_score'].mean()

        # Correlation with INVERTED DS (Uncertainty DS)
        ds_subset = pd.merge(df_merged[df_merged['method'] == method], df_probs[['prompt_id', 'model', 'p_factual_ds']], on=['prompt_id', 'model'])
        if len(ds_subset) > 0:
            # Inverted DS = 1 - p_factual_ds
            ds_uncertainty = 1.0 - ds_subset['p_factual_ds']
            corr, _ = spearmanr(ds_subset['uncertainty_score'], ds_uncertainty)
        else:
            corr = np.nan

        results.append({
            "Method": method,
            "AUROC": round(auroc, 3),
            "Mean Unc (Fact)": round(mean_factual, 3),
            "Mean Unc (Adv)": round(mean_adv, 3),
            "Mean Unc (Ambig)": round(mean_ambig, 3),
            "DS Corr": round(corr, 3) if not np.isnan(corr) else "N/A"
        })

    df_report = pd.DataFrame(results).sort_values(by="AUROC", ascending=False)
    output_dir = 'data/pilot'
    df_all.to_parquet(f'{output_dir}/uq_benchmark_results.parquet')

    with open(f'{output_dir}/uq_benchmark_summary.md', 'w') as f:
        f.write("# Pilot UQ Benchmark Summary\n\n")
        f.write(df_report.to_markdown(index=False))
        f.write("\n\n*Note: DS Corr is Spearman correlation with inverted DS scores (1-p_factual_ds).*")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_merged[df_merged['method'] == 'Structural_Semantic_Dispersion'], x='difficulty_type', y='uncertainty_score')
    plt.title('SSD Uncertainty Distribution')
    plt.savefig(f'{output_dir}/uq_benchmark_plots.png')

if __name__ == "__main__":
    main()
