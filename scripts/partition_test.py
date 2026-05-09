import pandas as pd
import numpy as np

def main():
    df_p = pd.read_parquet('data/pilot/pilot_100_prompts.parquet')

    # Exclude ambiguous for AUROC metrics as per instructions
    df_eval = df_p[df_p['difficulty_type'].isin(['factual', 'adversarial'])].copy()

    # 30 prompts total: 15 factual, 15 adversarial
    test_set = df_eval.groupby('difficulty_type').sample(n=15, random_state=42)
    test_ids = test_set['prompt_id'].tolist()

    with open('test_ids.txt', 'w') as f:
        f.write("\n".join(test_ids))

    print(f"✅ Partitioned {len(test_ids)} test prompts.")
    print(test_set['difficulty_type'].value_counts())

if __name__ == "__main__":
    main()
