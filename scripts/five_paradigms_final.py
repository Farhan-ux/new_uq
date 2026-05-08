import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from ripser import ripser
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import scipy.stats
import scipy.special

class ParadigmSuite:
    def __init__(self, device="cpu"):
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    def get_context(self, resps):
        clean = [r for r in resps if r.strip()]
        if len(clean) < 4: return None
        embs = self.embed_model.encode(clean)
        sims = cosine_similarity(embs)[0]
        pca = PCA(n_components=min(len(clean)-1, 10))
        emb_red = pca.fit_transform(embs)
        res_tda = ripser(emb_red, maxdim=0)
        h0 = res_tda['dgms'][0][np.isfinite(res_tda['dgms'][0][:, 1])][:, 1]
        h0_max = np.max(h0) if len(h0) > 0 else 0
        evs = pca.explained_variance_
        pr = (np.sum(evs)**2) / (np.sum(evs**2) + 1e-9)
        return sims, h0_max, pr, len(clean)

    # 1. Fuzzy Logic (Fuzzy Consensus)
    def method_fuzzy(self, sims, h0, pr, n):
        # Membership based on similarity to medoid
        mu = np.mean(scipy.special.expit(15 * (sims - 0.7)))
        return mu

    # 2. Interval Probabilities (P-Box Center)
    def method_interval(self, sims, h0, pr, n):
        # Uncertainty interval [mean-std, mean+std]
        mu = np.mean(sims)
        sigma = np.std(sims)
        return (mu + (mu + sigma)) / 2 # Upper-biased center

    # 3. Dempster-Shafer (Belief Function)
    def method_ds(self, sims, h0, pr, n):
        # Evidence combination
        m_factual = np.clip(sims[1:], 0, 0.9)
        belief = 1 - np.prod(1 - m_factual)
        return belief

    # 4. Possibility Theory (Necessity Measure)
    def method_possibility(self, sims, h0, pr, n):
        # Truth is necessary if all alternatives are impossible
        poss = np.max(sims[1:])
        necc = 1 - np.max(1 - sims[1:]) # min(sims)
        return (poss + necc) / 2

    # 5. Bayesian (Recursive Posterior)
    def method_bayesian(self, sims, h0, pr, n):
        p = 0.5
        for s in sims[1:]:
            l = scipy.special.expit(20 * (s - 0.75))
            p = (p * l) / (p * l + (1-p)*(1-l) + 1e-9)
        return p

def run():
    df_responses = pd.read_parquet('data/pilot/responses_100/responses.parquet')
    df_prompts = pd.read_parquet('data/pilot/pilot_100_prompts.parquet')
    df_responses = df_responses[df_responses['model'] != 'llama-3.3-70b-versatile'].copy()
    label_map = {'factual': 1, 'adversarial': 0}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label']], on='prompt_id')
    df = df[df['label'].isin([0, 1])].copy()
    
    suite = ParadigmSuite()
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        ctx = suite.get_context(row['responses'])
        if ctx:
            results.append({
                'label': row['label'],
                'Fuzzy': suite.method_fuzzy(*ctx),
                'Interval': suite.method_interval(*ctx),
                'DS': suite.method_ds(*ctx),
                'Possibility': suite.method_possibility(*ctx),
                'Bayesian': suite.method_bayesian(*ctx)
            })
    
    df_res = pd.DataFrame(results)
    print("\n--- Paradigm Comparison AUROC ---")
    for m in ['Fuzzy', 'Interval', 'DS', 'Possibility', 'Bayesian']:
        auc = roc_auc_score(df_res['label'], df_res[m])
        print(f"{m:12}: {auc:.4f}")

if __name__ == "__main__":
    run()
