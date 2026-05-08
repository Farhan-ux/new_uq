import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from ripser import ripser

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_raw_signals(resps):
    clean = [r for r in resps if r.strip()]
    embs = embed_model.encode(clean)
    n_comp = min(len(clean) - 1, 10)
    pca = PCA(n_components=n_comp)
    emb_reduced = pca.fit_transform(embs)
    evr = pca.explained_variance_ratio_
    id_proxy = np.sum(evr > 0.05)
    
    res_tda = ripser(emb_reduced, maxdim=0)
    h0 = res_tda['dgms'][0]
    h0_finite = h0[np.isfinite(h0[:, 1])]
    h_max = np.max(h0_finite[:, 1]) if len(h0_finite) > 0 else 0
    
    dist_matrix = euclidean_distances(embs)
    sigma = np.median(dist_matrix[np.triu_indices(len(clean), k=1)])
    if sigma == 0: sigma = 0.5
    kernel_matrix = np.exp(-(dist_matrix**2) / (2 * sigma**2))
    gammas = np.mean(kernel_matrix, axis=1)
    gamma_rel = gammas[0] / (np.median(gammas) + 1e-9)
    
    return h_max, gamma_rel, id_proxy

df_responses = pd.read_parquet('data/pilot/responses_100/responses.parquet')
target_prompt = "21965fbd-c711-4b29-958a-cd97ba1f66e2"
df_target = df_responses[df_responses['prompt_id'] == target_prompt]

for i, row in df_target.iterrows():
    h_max, g_rel, idp = get_raw_signals(row['responses'])
    print(f"Model: {row['model']}")
    print(f"H_max: {h_max:.4f}, G_rel: {g_rel:.4f}, ID_proxy: {idp}")
    print(f"SAME score component (H*G)/(ID+1): {(h_max * g_rel)/(idp + 1):.4f}")
    # print sample responses lengths
    lengths = [len(r) for r in row['responses']]
    print(f"Lengths: {lengths}")
    print("-" * 20)
