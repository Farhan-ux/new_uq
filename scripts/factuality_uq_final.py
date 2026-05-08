import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from ripser import ripser
import scipy.special
import sys

class factuality_uq:
    """
    Semantic-Algorithmic Manifold Evidence (SAME)

    A black-box uncertainty quantification method for LLM factuality.
    Developed by Jules for the Autonomous Research Directive.
    """
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Auxiliary model for semantic embedding (black-box restriction maintained)
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def estimate_factuality(self, responses):
        """
        Calculates the probability that the first response in the ensemble is correct.

        Args:
            responses (list of str): 10 stochastic responses to a prompt.

        Returns:
            float: Calibrated probability P(factual) ∈ [0, 1].
        """
        clean = [r for r in responses if r.strip()]
        if len(clean) < 3:
            return 0.5

        # 1. Embeddings
        embs = self.embed_model.encode(clean)

        # 2. PCA & Intrinsic Dimensionality Proxy
        n_comp = min(len(clean) - 1, 10)
        pca = PCA(n_components=n_comp)
        emb_reduced = pca.fit_transform(embs)
        id_proxy = np.sum(pca.explained_variance_ratio_ > 0.05)

        # 3. Persistent Homology (Topological Persistence)
        res_tda = ripser(emb_reduced, maxdim=0)
        h0 = res_tda['dgms'][0]
        h0_finite = h0[np.isfinite(h0[:, 1])]
        h_max = np.max(h0_finite[:, 1]) if len(h0_finite) > 0 else 0

        # 4. Spectral Centrality (Heat Kernel)
        dist_matrix = euclidean_distances(embs)
        triu_idx = np.triu_indices(len(clean), k=1)
        sigma = np.median(dist_matrix[triu_idx])
        if sigma == 0: sigma = 0.5

        kernel_matrix = np.exp(-(dist_matrix**2) / (2 * sigma**2))
        gammas = np.mean(kernel_matrix, axis=1)
        gamma_rel = gammas[0] / (np.median(gammas) + 1e-9)

        # 5. SAME Scoring & Sigmoid Calibration
        # Signal incorporates persistence, relative centrality, and dimensionality penalty.
        same_score = (h_max * gamma_rel) / (id_proxy + 1)

        # 6. Calibrated Probabilistic Mapping
        # k=5.8385, b=-0.4740 optimized to minimize ECE on 100-prompt benchmark.
        p_factual = float(scipy.special.expit(5.8385 * same_score - 0.4740))

        return p_factual

if __name__ == "__main__":
    # Minimal Example
    uq = factuality_uq()
    test_responses = [
        "The capital of France is Paris.",
        "Paris is the capital of France.",
        "The capital city of France is Paris.",
        "France's capital is Paris.",
        "Paris.",
        "It's Paris.",
        "The capital of France is Lyon.", # Hallucination 1
        "I believe it is Paris.",
        "Paris is the seat of the French government.",
        "The capital of France is Marseille." # Hallucination 2
    ]
    prob = uq.estimate_factuality(test_responses)
    print(f"Ensemble factuality probability: {prob:.2%}")
