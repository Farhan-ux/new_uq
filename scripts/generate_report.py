import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve

def generate_report():
    df = pd.read_parquet('data/pilot/uq_benchmark_results.parquet')
    df_prompts = pd.read_parquet('data/pilot/pilot_prompts_20.parquet')

    label_map = {'factual': 1, 'adversarial': 0, 'ambiguous': -1}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)

    df = pd.merge(df, df_prompts[['prompt_id', 'label', 'difficulty_type']], on='prompt_id')
    df_eval = df[df['label'].isin([0, 1])]

    y_true = df_eval['label'].values
    y_prob = df_eval['confidence_score'].values

    # 1. AUROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('TME Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # 2. Reliability Diagram
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5)

    plt.subplot(1, 2, 2)
    plt.plot(prob_pred, prob_true, "s-", label="TME")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted probability")
    plt.title("TME Reliability Diagram")
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig('tme_evaluation_report.png')
    print(f"Report saved to tme_evaluation_report.png. AUROC: {roc_auc:.3f}")

if __name__ == "__main__":
    generate_report()
