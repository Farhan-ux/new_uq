import numpy as np
import pandas as pd
import zlib
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from ripser import ripser
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import scipy.special

def get_lexical_redundancy(resps):
    try:
        ind_sizes = [len(zlib.compress(r.encode())) for r in resps]
        joint_size = len(zlib.compress(" ".join(resps).encode()))
        return np.sum(ind_sizes) / (joint_size + 1e-9)
    except: return 1.0

class MEC_Engine:
    def __init__(self, device="cpu"):
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    def compute_mec(self, resps):
        clean = [r for r in resps if r.strip()]
        n = len(clean)
        if n < 4: return 0.5
        embs = self.embed_model.encode(clean)
        
        # 1. Topological Evidence
        pca = PCA(n_components=min(n-1, 10))
        emb_red = pca.fit_transform(embs)
        res_tda = ripser(emb_red, maxdim=1)
        h0_max = np.max(res_tda['dgms'][0][np.isfinite(res_tda['dgms'][0][:, 1])][:, 1]) if len(res_tda['dgms'][0][np.isfinite(res_tda['dgms'][0][:, 1])]) > 0 else 0
        h1_max = np.max(res_tda['dgms'][1][:, 1] - res_tda['dgms'][1][:, 0]) if len(res_tda['dgms'][1]) > 0 else 0
        topo_signal = h0_max / (1.0 + h1_max)
        
        # 2. Bayesian Evidence (recursive)
        sims = cosine_similarity(embs)[0]
        p_bayes = 0.5
        for s in sims[1:]:
            l_f = 1 / (1 + np.exp(-15 * (s - 0.75)))
            p_bayes = (p_bayes * l_f) / (p_bayes * l_f + (1-p_bayes)*(1-l_f) + 1e-9)
            
        # 3. Lexical Redundancy
        redundancy = get_lexical_redundancy(clean)
        
        # MEC Final Formula: The "Truth Syllogism"
        # Truth is structurally robust AND semantically supported AND algorithmically redundant.
        score = np.log1p(topo_signal) * p_bayes * redundancy
        return score

def run():
    df_responses = pd.read_parquet('data/pilot/responses_100/responses.parquet')
    df_prompts = pd.read_parquet('data/pilot/pilot_100_prompts.parquet')
    df_responses = df_responses[df_responses['model'] != 'llama-3.3-70b-versatile'].copy()
    label_map = {'factual': 1, 'adversarial': 0}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label']], on='prompt_id')
    df = df[df['label'].isin([0, 1])].copy()
    
    uq = MEC_Engine(device="cpu")
    scores, labels, models = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        s = uq.compute_mec(row['responses'])
        scores.append(s); labels.append(row['label']); models.append(row['model'])
    
    print(f"\nMeta-Evidential Consensus (MEC) AUROC: {roc_auc_score(labels, scores):.4f}")
    df_res = pd.DataFrame({'label': labels, 'score': scores, 'model': models})
    for m in df_res['model'].unique():
        print(f"Model {m} AUROC: {roc_auc_score(df_res[df_res['model']==m]['label'], df_res[df_res['model']==m]['score']):.4f}")

if __name__ == "__main__":
    run()
