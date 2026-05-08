import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from ripser import ripser
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

class TNC_Engine:
    def __init__(self, device="cpu"):
        self.device = device
        self.nli_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-small")
        self.nli_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-small").to(self.device)
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def get_nli_centrality(self, resps):
        n = len(resps)
        pairs = []
        for i in range(n):
            for j in range(n):
                pairs.append((resps[i], resps[j]))

        batch_size = 128
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
        # NLI Score = prob(entailment) - prob(contradiction)
        scores = probs[:, 0] - probs[:, 2]
        mat = scores.reshape(n, n)
        # Adj matrix (thresholded for graph centrality)
        adj = (mat + 1.0) / 2.0 # Scale to [0, 1]
        # Eigenvector centrality (power iteration)
        v = np.ones(n) / n
        for _ in range(10):
            v = np.dot(adj, v)
            v = v / (np.linalg.norm(v) + 1e-9)
        return v[0] # Centrality of r1

    def compute_tnc(self, resps):
        clean = [r for r in resps if r.strip()]
        if len(clean) < 4: return None

        # 1. TDA & Spectral
        embs = self.embed_model.encode(clean)
        pca = PCA(n_components=min(len(clean)-1, 10))
        emb_red = pca.fit_transform(embs)
        res_tda = ripser(emb_red, maxdim=1)
        h0_max = np.max(res_tda['dgms'][0][np.isfinite(res_tda['dgms'][0][:, 1])][:, 1]) if len(res_tda['dgms'][0][np.isfinite(res_tda['dgms'][0][:, 1])]) > 0 else 0
        h1_max = np.max(res_tda['dgms'][1][:, 1] - res_tda['dgms'][1][:, 0]) if len(res_tda['dgms'][1]) > 0 else 0

        evs = pca.explained_variance_
        pr = (np.sum(evs)**2) / (np.sum(evs**2) + 1e-9)

        # 2. NLI Centrality
        nli_cent = self.get_nli_centrality(clean)

        # TNC Formula:
        # (Topological Stability * NLI Consensus) / Spectral Dimension
        score = (h0_max * nli_cent) / ( (pr**0.5) * (1.0 + h1_max) )
        return score

def run():
    df_responses = pd.read_parquet('data/pilot/responses_100/responses.parquet')
    df_prompts = pd.read_parquet('data/pilot/pilot_100_prompts.parquet')
    df_responses = df_responses[df_responses['model'] != 'llama-3.3-70b-versatile'].copy()
    label_map = {'factual': 1, 'adversarial': 0}
    df_prompts['label'] = df_prompts['difficulty_type'].map(label_map)
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'label']], on='prompt_id')
    df = df[df['label'].isin([0, 1])].copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = TNC_Engine(device=device)
    scores, labels, models = [], [], []
    for _, row in tqdm(df.head(30).iterrows(), total=30):
        s = engine.compute_tnc(row['responses'])
        if s is not None:
            scores.append(s); labels.append(row['label']); models.append(row['model'])

    print(f"\nOverall TNC AUROC: {roc_auc_score(labels, scores):.4f}")
    df_res = pd.DataFrame({'label': labels, 'score': scores, 'model': models})
    for m in df_res['model'].unique():
        print(f"Model {m} AUROC: {roc_auc_score(df_res[df_res['model']==m]['label'], df_res[df_res['model']==m]['score']):.4f}")

if __name__ == "__main__":
    run()
