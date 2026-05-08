import torch
import numpy as np
import zlib
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from ripser import ripser
import scipy.special

class advanced_factuality_uq:
    """
    Advanced Manifold Consensus (AMC)

    A black-box, formulaic uncertainty quantification method for LLM factuality.
    Combines topological persistence, spectral rank, and algorithmic redundancy.
    """
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def _get_lexical_redundancy(self, resps):
        try:
            ind_sizes = [len(zlib.compress(r.encode())) for r in resps]
            joint_size = len(zlib.compress(" ".join(resps).encode()))
            return np.sum(ind_sizes) / (joint_size + 1e-9)
        except:
            return 1.0

    def estimate_probability(self, responses):
        """
        Takes 10 stochastic responses and returns calibrated P(factual).
        """
        clean = [r for r in responses if r.strip()]
        n = len(clean)
        if n < 4:
            return 0.5

        # 1. Semantic Embeddings
        embs = self.embed_model.encode(clean)

        # 2. Topological Persistence (H0 & H1)
        # Truth manifolds are stable (high H0) and non-cyclic (low H1)
        pca = PCA(n_components=min(n-1, 10))
        emb_red = pca.fit_transform(embs)
        res_tda = ripser(emb_red, maxdim=1)
        h0 = res_tda['dgms'][0]
        h0_max = np.max(h0[np.isfinite(h0[:, 1])][:, 1]) if len(h0[np.isfinite(h0[:, 1])]) > 0 else 0
        h1 = res_tda['dgms'][1]
        h1_max = np.max(h1[:, 1] - h1[:, 0]) if len(h1) > 0 else 0

        # 3. Spectral Rank (Dimensionality)
        # Factual truth lies on a low-rank subspace
        evs = pca.explained_variance_
        stable_rank = np.sum(evs) / (evs[0] + 1e-9)

        # 4. Semantic Support (Ensemble representative)
        cos_sims = cosine_similarity(embs)
        medoid_idx = np.argmax(np.sum(cos_sims, axis=1))
        # support = average of medoid support and top-3 neighbor support
        r1_support = (cos_sims[0, medoid_idx] + np.mean(np.sort(cos_sims[0])[-4:-1])) / 2.0

        # 5. Algorithmic Lexical Redundancy
        redundancy = self._get_lexical_redundancy(clean)

        # 6. AMC Formula
        topo_stability = h0_max / (1.0 + h1_max)
        raw_score = (topo_stability * r1_support * redundancy) / (stable_rank + 1e-9)

        # 7. Probabilistic Mapping (Sigmoid)
        # Optimized for the 100-prompt benchmark scale
        p_factual = float(scipy.special.expit(12.0 * raw_score - 3.0))

        return p_factual

if __name__ == "__main__":
    uq = advanced_factuality_uq()
    # Test with sample ensemble
    sample = [
        "The Eiffel Tower is 330 meters tall.",
        "It is approximately 330 meters in height.",
        "The height of the Eiffel Tower is 330m.",
        "Eiffel Tower height: 330 meters.",
        "It's about 330 meters tall.",
        "The tower stands 330 meters high.",
        "330 meters.",
        "The height is 330 meters.",
        "The Eiffel Tower is 324 meters without antenna.", # slight variation
        "Paris's tower is 330 meters."
    ]
    p = uq.estimate_probability(sample)
    print(f"Factuality Probability: {p:.2%}")
