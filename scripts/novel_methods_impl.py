import numpy as np
import zlib
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import entropy
from itertools import combinations

class NovelUQMethods:
    def __init__(self, device="cpu"):
        self.device = device
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def get_embeddings(self, resps):
        clean = [r for r in resps if r.strip()]
        if len(clean) < 3: return None, None
        return self.embed_model.encode(clean), clean

    def method_bayesian_gmm(self, embs):
        """Bayesian Model Evidence via BIC of GMMs"""
        if embs is None or len(embs) < 3: return 0.5
        # Project to 2D for stability
        pca = PCA(n_components=min(2, len(embs)-1))
        x_2d = pca.fit_transform(embs)

        bics = []
        for k in [1, 2]:
            try:
                gmm = GaussianMixture(n_components=k, reg_covar=1e-4, random_state=42)
                gmm.fit(x_2d)
                bics.append(gmm.bic(x_2d))
            except:
                bics.append(1e10)

        # Lower BIC is better. Evidence Ratio approx exp(delta_BIC/2)
        delta = bics[1] - bics[0] # Positive means 1-cluster is better
        prob = 1 / (1 + np.exp(-delta/2))
        return np.clip(prob, 0.01, 0.99)

    def method_dempster_shafer(self, embs):
        """Evidence Theory: Mass assignment based on semantic agreement"""
        if embs is None or len(embs) < 3: return 0.5
        sim_mat = cosine_similarity(embs)
        # Mass of agreement (Factual)
        m_factual = np.mean(sim_mat[np.triu_indices(len(embs), k=1)])
        # Mass of conflict (Non-Factual)
        m_conflict = 1.0 - m_factual
        # Simple Belief: Bel(F) = m(F)
        return np.clip(m_factual, 0.01, 0.99)

    def method_fuzzy_entropy(self, embs):
        """Fuzzy Sets: Membership entropy on the semantic manifold"""
        if embs is None or len(embs) < 3: return 0.5
        dists = euclidean_distances(embs)
        # Membership to the 'consensus' set
        sigma = np.median(dists) + 1e-9
        mu = np.exp(- (dists**2) / (2 * sigma**2))
        avg_mu = np.mean(mu, axis=1)
        # Fuzzy Entropy: -sum(mu*log(mu) + (1-mu)*log(1-mu))
        ent = -np.mean(avg_mu * np.log(avg_mu + 1e-9) + (1 - avg_mu) * np.log(1 - avg_mu + 1e-9))
        # Higher entropy -> Higher uncertainty. Return 1 - norm_ent
        return np.clip(1.0 - ent, 0.01, 0.99)

    def method_pbox_interval(self, embs):
        """Interval Probabilities: P-Box area via bootstrapping"""
        if embs is None or len(embs) < 4: return 0.5
        # Use average cosine similarity to medoid as the base statistic
        sims = cosine_similarity(embs)
        medoid_idx = np.argmax(np.sum(sims, axis=1))
        scores = sims[medoid_idx]

        # Bootstrap to find upper and lower CDFs
        boot_means = []
        for _ in range(50):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            boot_means.append(np.mean(sample))

        # Uncertainty interval (P-Box width)
        lower, upper = np.percentile(boot_means, [5, 95])
        interval_width = upper - lower
        return np.clip(1.0 - interval_width, 0.01, 0.99)

    def method_possibility_theory(self, embs):
        """Possibility Theory: Necessity Measure"""
        if embs is None or len(embs) < 3: return 0.5
        sims = cosine_similarity(embs)
        # Possibility distribution pi(x) = sim(x, medoid)
        medoid_idx = np.argmax(np.sum(sims, axis=1))
        pi = sims[medoid_idx]

        # Necessity N(A) = 1 - sup_{x not in A} pi(x)
        # Here we define A as the set of responses close to the medoid.
        # We can simplify: N = mean(pi) * (1 - spread)
        necessity = np.mean(pi) * (1.0 - np.std(pi))
        return np.clip(necessity, 0.01, 0.99)

    def compute_all(self, resps):
        embs, clean = self.get_embeddings(resps)
        if embs is None: return None

        return {
            "Bayesian_GMM": self.method_bayesian_gmm(embs),
            "Dempster_Shafer": self.method_dempster_shafer(embs),
            "Fuzzy_Entropy": self.method_fuzzy_entropy(embs),
            "PBox_Interval": self.method_pbox_interval(embs),
            "Possibility_Necessity": self.method_possibility_theory(embs)
        }
