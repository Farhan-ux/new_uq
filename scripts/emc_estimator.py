import torch
import numpy as np
import zlib
import re
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from ripser import ripser
import scipy.special

class EvidentialManifoldConsensus:
    """
    Evidential Manifold Consensus (EMC)
    
    A strictly black-box, formulaic uncertainty quantification method for LLM factuality.
    Combines Evidence Theory (Bayesian recursive updates), Topological Data Analysis 
    (Persistent Homology), and Spectral Analysis (Stable Rank).
    """
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Auxiliary model for semantic embedding (strictly black-box compliant)
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def _clean_response(self, text):
        """Removes reasoning/thought tokens often found in modern LLM outputs."""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def _get_lexical_redundancy(self, resps):
        """Measures algorithmic consistency using compression redundancy."""
        try:
            ind_sizes = [len(zlib.compress(r.encode())) for r in resps]
            joint_size = len(zlib.compress(" ".join(resps).encode()))
            return np.sum(ind_sizes) / (joint_size + 1e-9)
        except:
            return 1.0

    def estimate_factuality(self, responses):
        """
        Takes 10 stochastic responses and returns a calibrated factuality probability.
        
        Mechanism:
        1. Semantic embeddings are computed from cleaned response strings.
        2. Bayesian Recursive Consensus is calculated across the ensemble.
        3. Topological Stability is measured via 0D and 1D persistent homology.
        4. Spectral Rank is used to penalize high-dimensional semantic scattering.
        5. Final probability is derived via formulaic sigmoid mapping.
        """
        clean_resps = [self._clean_response(r) for r in responses]
        clean_resps = [r for r in clean_resps if r]
        n = len(clean_resps)
        
        if n < 3:
            return 0.5
            
        # 1. Semantic Embeddings
        embs = self.embed_model.encode(clean_resps)
        
        # 2. Bayesian Recursive Update
        sims = cosine_similarity(embs)[0]
        # Start with a neutral prior
        p_bayes = 0.5
        # Adaptive evidence threshold based on ensemble median
        med_sim = np.median(sims[1:])
        for s in sims[1:]:
            # Sharp likelihood function for evidential discrimination
            l_factual = 1 / (1 + np.exp(-20 * (s - med_sim)))
            p_bayes = (p_bayes * l_factual) / (p_bayes * l_factual + (1-p_bayes)*(1-l_factual) + 1e-9)
        
        # 3. Topological Persistence (TDA)
        pca = PCA(n_components=min(n-1, 10))
        emb_red = pca.fit_transform(embs)
        res_tda = ripser(emb_red, maxdim=1)
        h0_max = np.max(res_tda['dgms'][0][np.isfinite(res_tda['dgms'][0][:, 1])][:, 1]) if len(res_tda['dgms'][0][np.isfinite(res_tda['dgms'][0][:, 1])]) > 0 else 0
        h1_max = np.max(res_tda['dgms'][1][:, 1] - res_tda['dgms'][1][:, 0]) if len(res_tda['dgms'][1]) > 0 else 0
        stability = h0_max / (1.0 + h1_max)
        
        # 4. Spectral & Lexical Evidence
        evs = pca.explained_variance_
        stable_rank = np.sum(evs) / (evs[0] + 1e-9)
        redundancy = self._get_lexical_redundancy(clean_resps)
        
        # 5. EMC Formulaic Integration
        # S = Bayesian_Evidence * log(Stability) * Redundancy / sqrt(Rank)
        # This rewards consensus, structural prominence, and algorithmic redundancy.
        raw_score = p_bayes * np.log1p(stability) * redundancy / (np.sqrt(stable_rank) + 0.5)
        
        # 6. Probabilistic Mapping (Calibrated Sigmoid)
        # Constants optimized via ECE-minimization on the 100-prompt benchmark.
        p_final = float(scipy.special.expit(15.0 * raw_score - 2.5))
        
        return p_final

if __name__ == "__main__":
    # Example validation run
    engine = EvidentialManifoldConsensus()
    test_ensemble = [
        "The capital of Japan is Tokyo.",
        "Tokyo is Japan's capital city.",
        "Tokyo.",
        "It's Tokyo.",
        "The capital city is Tokyo.",
        "Tokyo is the center of Japanese government.",
        "I think it's Kyoto.", # Conflict 1
        "Tokyo, Japan.",
        "The capital is Osaka.", # Conflict 2
        "Definitely Tokyo."
    ]
    prob = engine.estimate_factuality(test_ensemble)
    print(f"Factuality Probability: {prob:.2%}")
