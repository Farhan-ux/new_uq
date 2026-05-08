import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from ripser import ripser
from tqdm import tqdm

df_metrics = pd.read_parquet('final_experiment_100_full.parquet')
df_eval = df_metrics[df_metrics['label'].isin([0, 1])].copy()

# Since we don't have the raw components in the parquet, we might need to re-extract them for a sample or all
# Or we can just look at the stats of SAME vs label
print(df_eval.groupby('label')[['SAME', 'iTME']].describe())

# Re-run for a subset to get raw components if needed, but let's try to find more failures first
fp = df_eval[(df_eval['label'] == 0)].sort_values('SAME', ascending=False).head(5)
print("\n--- Top False Positives (Adversarial with high scores) ---")
print(fp[['prompt_id', 'model', 'SAME', 'label']])

fn = df_eval[(df_eval['label'] == 1)].sort_values('SAME', ascending=True).head(5)
print("\n--- Top False Negatives (Factual with low scores) ---")
print(fn[['prompt_id', 'model', 'SAME', 'label']])
