import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from ripser import ripser

class Llama3UQ:
    """
    Architecture-Aware UQ for Llama-3.1-8B-Instant.
    Optimization: Graph-Topological Consensus.
    Logic: Llama-3's RLHF training enforces high semantic alignment for factual truths.
    """
    def __init__(self, device="cpu"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    def compute(self, resps):
        clean = [r for r in resps if r.strip()]
        if len(clean) < 3: return 0.5
        embs = self.model.encode(clean)

        # 1. Topological Persistence (H1) - Low H1 is Factual
        res_tda = ripser(embs, maxdim=1)
        h1 = res_tda['dgms'][1]
        h1_max = np.max(h1[:, 1] - h1[:, 0]) if len(h1) > 0 else 0
        s_h1 = np.clip(1.0 - h1_max, 0, 1)

        # 2. Graph Closeness - High Closeness is Factual
        sim_mat = cosine_similarity(embs)
        G = nx.Graph()
        for i in range(len(clean)):
            for j in range(i+1, len(clean)):
                if sim_mat[i, j] > 0.8: G.add_edge(i, j)
        closeness = np.mean(list(nx.closeness_centrality(G).values())) if len(G.edges) > 0 else 0

        # 3. Bayesian BIC - Evidence for a single mode
        pca = PCA(n_components=min(2, len(embs)-1))
        x_2d = pca.fit_transform(embs)
        bics = [GaussianMixture(n_components=k, reg_covar=1e-4).fit(x_2d).bic(x_2d) for k in [1, 2]]
        gmm_prob = 1 / (1 + np.exp(-(bics[1] - bics[0])/2))

        # Formula: ((1-h1) * Closeness) / ((1-GMM) + 0.1)
        score = (s_h1 * closeness) / ((1.0 - gmm_prob) + 0.1)
        return np.clip(score / 5.0, 0, 1) # Normalization constant

if __name__ == "__main__":
    uq = Llama3UQ()
    print("Score:", uq.compute(["Paris is in France.", "Paris is the French capital.", "The capital of France is Paris."]))
