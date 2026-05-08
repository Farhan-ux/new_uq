import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from ripser import ripser
import scipy.special

class SAME_UQ:
    """
    Semantic-Algorithmic Manifold Evidence (SAME)

    A black-box uncertainty quantification method for LLM factuality.
    Uses 0D persistent homology, heat-kernel centrality, and
    intrinsic dimensionality proxies to estimate truth probability.
    """
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Lightweight auxiliary model
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def predict_probability(self, responses):
        """
        Takes exactly 10 stochastic responses and returns P(factual) in [0, 1].
        """
        clean = [r for r in responses if r.strip()]
        if len(clean) < 3:
            return 0.5

        # 1. Generate Embeddings
        embs = self.embed_model.encode(clean)

        # 2. Intrinsic Dimensionality Proxy (via PCA)
        # Truth manifolds typically lie on lower-dimensional subspaces
        n_comp = min(len(clean) - 1, 10)
        pca = PCA(n_components=n_comp)
        emb_reduced = pca.fit_transform(embs)
        evr = pca.explained_variance_ratio_
        id_proxy = np.sum(evr > 0.05) # Count components explaining >5% variance

        # 3. Topological Persistence (0D Persistent Homology)
        # H_max captures the 'semantic diameter' of the primary consensus cluster.
        res_tda = ripser(emb_reduced, maxdim=0)
        h0 = res_tda['dgms'][0]
        h0_finite = h0[np.isfinite(h0[:, 1])]
        h_max = np.max(h0_finite[:, 1]) if len(h0_finite) > 0 else 0

        # 4. Heat Kernel Centrality (Evidential Support)
        # Quantifies how centrally r1 (the primary response) is located.
        dist_matrix = euclidean_distances(embs)
        triu_idx = np.triu_indices(len(clean), k=1)
        sigma = np.median(dist_matrix[triu_idx]) # Adaptive bandwidth
        if sigma == 0: sigma = 0.5

        kernel_matrix = np.exp(-(dist_matrix**2) / (2 * sigma**2))
        gammas = np.mean(kernel_matrix, axis=1)
        gamma_r1 = gammas[0]
        gamma_median = np.median(gammas)

        # 5. SAME Scoring Function
        # We reward high persistence (H_max) and high relative centrality (gamma_r1 / gamma_med)
        # while penalizing high intrinsic dimensionality (id_proxy).
        same_score = (h_max * (gamma_r1 / (gamma_median + 1e-9))) / (id_proxy + 1)

        # 6. Probabilistic Mapping (Sigmoid Transform)
        # Calibrated based on pilot performance data.
        p_factual = float(scipy.special.expit(4.0 * same_score - 2.0))

        return p_factual

# Example Usage:
# uq = SAME_UQ()
# p = uq.predict_probability(["Paris is the capital of France.", ...])
# print(f"Estimated Probability of Correctness: {p:.2%}")
