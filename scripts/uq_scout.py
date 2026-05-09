import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from ripser import ripser

class ScoutUQ:
    """
    Architecture-Aware UQ for Llama-4-Scout (MoE Architecture).
    Optimization: Spectral Rank & Fuzzy Manifold Persistence.
    Logic: Scout's MoE gating creates distinct spectral signatures for factual vs hallucinatory paths.
    """
    def __init__(self, device="cpu"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    def compute(self, resps):
        clean = [r for r in resps if r.strip()]
        if len(clean) < 3: return 0.5
        embs = self.model.encode(clean)

        # 1. Topological Persistence (H1)
        res_tda = ripser(embs, maxdim=1)
        h1 = res_tda['dgms'][1]
        h1_max = np.max(h1[:, 1] - h1[:, 0]) if len(h1) > 0 else 0
        s_h1 = np.clip(1.0 - h1_max, 0, 1)

        # 2. Fuzzy Entropy
        dists = euclidean_distances(embs)
        sigma = np.median(dists) + 1e-9
        mu = np.exp(- (dists**2) / (2 * sigma**2))
        avg_mu = np.mean(mu, axis=1)
        fuzzy_ent = -np.mean(avg_mu * np.log(avg_mu + 1e-9) + (1 - avg_mu) * np.log(1 - avg_mu + 1e-9))
        s_fuzzy = np.clip(1.0 - fuzzy_ent, 0, 1)

        # 3. Stable Rank
        singular_values = PCA().fit(embs).singular_values_
        stable_rank = np.sum(singular_values**2) / (np.max(singular_values)**2 + 1e-9)

        # Formula: ((1-h1) * (1-Fuzzy)) / (Rank + 0.1) -> Adjusted: (s_h1 * s_fuzzy) / (stable_rank + 0.1)
        score = (s_h1 * s_fuzzy) / (stable_rank + 0.1)
        return np.clip(score / 2.0, 0, 1)

if __name__ == "__main__":
    uq = ScoutUQ()
    print("Score:", uq.compute(["Paris is in France.", "Paris is the French capital.", "The capital of France is Paris."]))
