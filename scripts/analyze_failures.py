import pandas as pd
import numpy as np

df_metrics = pd.read_parquet('final_experiment_100_full.parquet')
df_responses = pd.read_parquet('data/pilot/responses_100/responses.parquet')
df = pd.merge(df_metrics, df_responses, on=['prompt_id', 'model'])

# Look for False Negatives (Factual but low SAME score)
fn = df[(df['label'] == 1) & (df['SAME'] < 0.2)].head(3)
print("--- False Negatives (Factual but low score) ---")
for i, row in fn.iterrows():
    print(f"Prompt: {row['prompt_id']}, Model: {row['model']}, SAME: {row['SAME']:.4f}")
    print(f"Sample Resp: {row['responses'][0][:100]}...")
    print("-" * 20)

# Look for False Positives (Adversarial but high SAME score)
fp = df[(df['label'] == 0) & (df['SAME'] > 0.8)].head(3)
print("\n--- False Positives (Adversarial but high score) ---")
for i, row in fp.iterrows():
    print(f"Prompt: {row['prompt_id']}, Model: {row['model']}, SAME: {row['SAME']:.4f}")
    print(f"Sample Resp: {row['responses'][0][:100]}...")
    print("-" * 20)
