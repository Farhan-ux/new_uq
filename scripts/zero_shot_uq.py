import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from ripser import ripser

def zero_shot_prob(x, midpoint, k=4.0):
    """Principled sigmoid mapping for zero-shot probability estimation."""
    return 1 / (1 + np.exp(-k * (x - midpoint)))

class BaseZeroShotUQ:
    def __init__(self, device="cpu"):
        self.device = device
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def get_features(self, resps):
        clean = [r for r in resps if r.strip()]
        if len(clean) < 3: return None
        embs = self.embed_model.encode(clean)
        pca = PCA().fit(embs)
        ev1_ratio = pca.explained_variance_[0] / (np.sum(pca.explained_variance_) + 1e-9)
        stable_rank = np.sum(pca.singular_values_**2) / (np.max(pca.singular_values_)**2 + 1e-9)
        res_tda = ripser(embs, maxdim=0)
        h0_max = np.max(res_tda['dgms'][0][np.isfinite(res_tda['dgms'][0][:, 1])][:, 1])
        return {
            'ev1_ratio': ev1_ratio,
            'stable_rank': stable_rank,
            'h0_max': h0_max
        }

class ZeroShotLlama3UQ(BaseZeroShotUQ):
    """Dense/RLHF: Spectral Dominance."""
    def compute(self, resps):
        f = self.get_features(resps)
        if not f: return 0.5
        # We use a very low k to prioritize low ECE (Expected Calibration Error)
        return zero_shot_prob(f['ev1_ratio'], midpoint=0.55, k=2.5)

class ZeroShotQwenUQ(BaseZeroShotUQ):
    """Diverse/Pretrained: Dimensional Density."""
    def compute(self, resps):
        f = self.get_features(resps)
        if not f: return 0.5
        score = f['h0_max'] / (f['stable_rank'] + 0.1)
        return zero_shot_prob(score, midpoint=0.15, k=6.0)

class ZeroShotScoutUQ(BaseZeroShotUQ):
    """MoE: Spectral Rank Stability."""
    def compute(self, resps):
        f = self.get_features(resps)
        if not f: return 0.5
        stability = 1.0 / (f['stable_rank'] + 1e-9)
        return zero_shot_prob(stability, midpoint=0.45, k=3.5)
