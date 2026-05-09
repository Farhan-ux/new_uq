import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from ripser import ripser
from sklearn.decomposition import PCA
import torch

class EMRFeatureExtractor:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def extract(self, resps):
        clean = [r for r in resps if r.strip()]
        if len(clean) < 3: return None

        embs = self.model.encode(clean)

        # 1. Topological Features
        res_tda = ripser(embs, maxdim=1)
        h0 = res_tda['dgms'][0]
        h0_max = np.max(h0[np.isfinite(h0[:, 1])][:, 1]) if len(h0[np.isfinite(h0[:, 1])]) > 0 else 0
        h1 = res_tda['dgms'][1]
        h1_max = np.max(h1[:, 1] - h1[:, 0]) if len(h1) > 0 else 0

        # 2. Spectral Features
        pca = PCA()
        pca.fit(embs)
        singular_values = pca.singular_values_
        stable_rank = np.sum(singular_values**2) / (np.max(singular_values)**2 + 1e-9)

        # 3. Entailment Graph Features
        # Simple cosine-based entailment proxy for graph construction
        sim_mat = embs @ embs.T / (np.linalg.norm(embs, axis=1)[:, None] @ np.linalg.norm(embs, axis=1)[None, :] + 1e-9)
        G = nx.DiGraph()
        for i in range(len(clean)):
            for j in range(len(clean)):
                if i != j and sim_mat[i, j] > 0.8:
                    G.add_edge(i, j)

        pr = nx.pagerank(G) if len(G.edges) > 0 else {i: 1/len(clean) for i in range(len(clean))}
        closeness = nx.closeness_centrality(G) if len(G.edges) > 0 else {i: 0 for i in range(len(clean))}

        # 4. Lexical Features
        raw = " ".join(clean)
        import zlib
        lex_red = len(raw) / (len(zlib.compress(raw.encode())) + 1e-9)

        return {
            'h0_max': h0_max,
            'h1_max': h1_max,
            'stable_rank': stable_rank,
            'page_rank': np.mean(list(pr.values())),
            'closeness': np.mean(list(closeness.values())),
            'lex_red': lex_red,
            'max_contra': 1.0 - np.mean(sim_mat), # Proxy for contradiction
            'entail_r1': np.mean(sim_mat > 0.8) # Proxy for entailment ratio
        }
