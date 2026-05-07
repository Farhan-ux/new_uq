import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import scipy.stats
import os
from tqdm import tqdm

class BaselineUQ:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nli_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-small")
        self.nli_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-small").to(self.device)
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def get_nli_matrix(self, resps):
        n = len(resps)
        if n == 0: return np.zeros((0,0))
        pairs = []
        for i in range(n):
            for j in range(n):
                pairs.append((resps[i], resps[j]))

        batch_size = 256 # Higher batch size
        all_probs = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            encoded = self.nli_tokenizer([p[0] for p in batch], [p[1] for p in batch],
                                       padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.nli_model(**encoded).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)

        probs = np.vstack(all_probs)
        scores = probs[:, 0] - probs[:, 2]
        return scores.reshape(n, n)

    def compute_metrics(self, resps):
        if not resps or len(resps) < 2:
            return {
                "Lexical_Similarity": 0.0, "Semantic_Entropy": 0.0, "NumSemSets": 1.0,
                "EigValLaplacian": 0.0, "DegMat": 0.0, "Eccentricity": 0.0, "Semantic_Density": 0.0
            }

        def jaccard(s1, s2):
            w1, w2 = set(s1.lower().split()), set(s2.lower().split())
            return len(w1 & w2) / len(w1 | w2) if (w1 | w2) else 0

        lex_sims = [jaccard(resps[i], resps[j]) for i in range(len(resps)) for j in range(i+1, len(resps))]
        lex_sim = np.mean(lex_sims)

        nli_mat = self.get_nli_matrix(resps)
        dist_mat = 1.0 - (nli_mat + 1.0) / 2.0
        np.fill_diagonal(dist_mat, 0)

        clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average', distance_threshold=0.5)
        clusters = clustering.fit_predict(dist_mat)

        num_sets = float(len(set(clusters)))
        counts = np.bincount(clusters)
        probs = counts / len(resps)
        sem_entropy = scipy.stats.entropy(probs)

        adj = (nli_mat > 0.0).astype(float)
        deg = np.sum(adj, axis=1)
        deg_mat = np.mean(deg)
        D = np.diag(deg)
        L = D - adj
        eigenvals = np.linalg.eigvalsh(L)
        sum_eig = np.sum(eigenvals)
        ecc = np.mean(np.max(dist_mat, axis=1))

        embs = self.embed_model.encode(resps)
        cos_sim = cosine_similarity(embs)
        sem_density = np.mean(cos_sim)

        return {
            "Lexical_Similarity": -lex_sim,
            "Semantic_Entropy": sem_entropy,
            "NumSemSets": num_sets,
            "EigValLaplacian": sum_eig,
            "DegMat": -deg_mat,
            "Eccentricity": ecc,
            "Semantic_Density": -sem_density
        }

def main():
    df_responses = pd.read_parquet('data/pilot/responses/pilot_responses_groq.parquet')
    uq = BaselineUQ()
    all_results = []
    for _, row in tqdm(df_responses.iterrows(), total=len(df_responses), desc="Baselines"):
        resps = [r for r in row['responses'] if r.strip()]
        metrics = uq.compute_metrics(resps)
        for m_name, val in metrics.items():
            all_results.append({
                "prompt_id": row['prompt_id'], "model": row['model'], "method": m_name,
                "uncertainty_score": float(val), "n_responses_used": len(resps)
            })
    pd.DataFrame(all_results).to_parquet('experiments/pilot_uq_benchmark/baseline_scores.parquet')

if __name__ == "__main__":
    main()
