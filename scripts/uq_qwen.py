import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class QwenUQ:
    """
    Architecture-Aware UQ for Qwen-32B.
    Optimization: Logical Consistency & Evidence Theory.
    Logic: Qwen's large-scale pretraining allows for diverse but logically consistent factuals.
    """
    def __init__(self, device="cpu"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    def compute(self, resps):
        clean = [r for r in resps if r.strip()]
        if len(clean) < 3: return 0.5
        embs = self.model.encode(clean)

        # 1. Contradiction Detection (Max Contra)
        sim_mat = cosine_similarity(embs)
        avg_sim = np.mean(sim_mat[np.triu_indices(len(clean), k=1)])
        s_contra = np.clip(avg_sim, 0, 1) # Higher is more consistent

        # 2. Dempster-Shafer Belief
        triu = sim_mat[np.triu_indices(len(clean), k=1)]
        belief = np.mean(triu > 0.8)

        # 3. Interval Probability (P-Box)
        medoid_sims = sim_mat[np.argmax(np.sum(sim_mat, axis=1))]
        boots = [np.mean(np.random.choice(medoid_sims, size=len(medoid_sims), replace=True)) for _ in range(30)]
        pbox_width = np.percentile(boots, 95) - np.percentile(boots, 5)

        # Formula: ((1-Contra) * DS) / (PBox + 0.1) -> Adjusted to: (S_Contra * Belief) / (PBox + 0.1)
        score = (s_contra * belief) / (pbox_width + 0.1)
        return np.clip(score / 5.0, 0, 1)

if __name__ == "__main__":
    uq = QwenUQ()
    print("Score:", uq.compute(["Paris is in France.", "Paris is the French capital.", "The capital of France is Paris."]))
