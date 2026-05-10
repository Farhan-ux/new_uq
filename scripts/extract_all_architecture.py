import pandas as pd
import numpy as np
import sys
import os
from tqdm import tqdm
import torch

# Import directly from the script file
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

fe_mod = load_module("feature_extractor", "scripts/feature_extractor.py")
EMRFeatureExtractor = fe_mod.EMRFeatureExtractor

uq_mod = load_module("final_novel_method", "scripts/final_novel_method.py")
ResearchBreakthroughUQ = uq_mod.ResearchBreakthroughUQ

def main():
    print("🚀 Extracting All Features for Architecture-Aware UQ...")

    try:
        df_responses = pd.read_parquet('data/pilot/responses_100/responses.parquet')
        df_prompts = pd.read_parquet('data/pilot/pilot_100_prompts.parquet')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Filter for target models
    target_models = [
        'llama-3.1-8b-instant',
        'qwen/qwen3-32b',
        'meta-llama/llama-4-scout-17b-16e-instruct'
    ]
    df_responses = df_responses[df_responses['model'].isin(target_models)].copy()

    label_map = {'factual': 1, 'adversarial': 0}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label']], on='prompt_id')
    df = df[df['label'].isin([0, 1])].copy()

    device = "cpu"
    fe = EMRFeatureExtractor(device=device)
    uq = ResearchBreakthroughUQ(device=device)

    all_data = []
    output_path = "arch_features_100.parquet"

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        resps = row['responses']
        try:
            # 1. EMR Features
            feats = fe.extract(resps)
            if not feats: continue

            # 2. Novel UQ Scores
            scores = uq.compute_ste_hybrid(resps)
            if scores:
                feats.update(scores)

            feats.update({
                'prompt_id': row['prompt_id'],
                'model': row['model'],
                'label': row['label']
            })
            all_data.append(feats)
        except Exception as e:
            # print(f"Skipping {idx}: {e}")
            pass

        if (len(all_data) > 0 and len(all_data) % 20 == 0) or idx == len(df)-1:
            pd.DataFrame(all_data).to_parquet(output_path)

    print(f"✅ Features saved to {output_path}")

if __name__ == "__main__":
    main()
