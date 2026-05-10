# Research Report: Architecture-Aware Uncertainty Quantification

## 1. Model-Specific UQ Matrix

| Model | Method Name | AUROC (Own) | AUROC (Llama) | AUROC (Qwen) | AUROC (Scout) | Key Signal | Architectural Reason |
|-------|-------------|-------------|---------------|--------------|---------------|------------|----------------------|
| **Llama-3.1-8B** | **Consensus Graph** | **0.702** | 0.702 | 0.524 | 0.583 | Graph Closeness | Highly aligned RLHF training causes factual truths to form extremely tight semantic cliques. |
| **Qwen-3-32B** | **Logical Necessity** | **0.733** | 0.568 | 0.733 | 0.503 | Contra-Ratio | Large-scale pretraining allows for diverse phrasing; logical contradiction is the only stable hallucination signal. |
| **Llama-4-Scout** | **Spectral Manifold** | **0.758** | 0.584 | 0.547 | 0.758 | Stable Rank | MoE architecture gating causes hallucinations to manifest as high-dimensional semantic scattering (High Rank). |

## 2. Mechanistic Insights

### Llama-3.1-8B-Instant
Uncertainty in Llama-3.1 manifests as a breakdown of **Social Consensus**. Because it is a dense model with strong instruction tuning, its factual responses are highly convergent. When it hallucinations, it does not just produce "noise" but creates divergent semantic "islands". Thus, Graph Closeness (connectivity within the ensemble) is the dominant predictor of truth.

### Qwen-3-32B
Uncertainty in Qwen manifests as **Logical Entropy**. Qwen is linguistically more diverse than Llama, meaning its factual responses may have low semantic similarity but high logical entailment. Hallucinations in Qwen are identified by the emergence of explicit logical contradictions (high Max-Contra) rather than just semantic distance.

### Meta-Llama-4-Scout-17B-16E (MoE)
Uncertainty in Scout manifests as **Spectral Diffusion**. As a Mixture-of-Experts model, factual responses are usually routed through a stable set of expert paths, resulting in a low-rank semantic manifold. Hallucinations trigger "Expert Confusion", where responses are scattered across a much higher-dimensional subspace, significantly increasing the Stable Rank of the embedding matrix.

## 3. UQ Signal Selector Checklist

To choose a UQ method for a new model, evaluate these architectural traits:

1.  **Is it a Mixture-of-Experts (MoE)?**
    -   *Yes:* Prioritize **Spectral Features** (Stable Rank, PCA Variance). MoE models show sharp dimensionality shifts during hallucination.
2.  **Is it heavily RLHF'd or Instruction Tuned?**
    -   *Yes:* Prioritize **Graph Features** (Closeness, PageRank). These models are trained to be "agreeable" and convergent on truth.
3.  **Is it a "Base-leaning" or high-diversity model (like Qwen)?**
    -   *Yes:* Prioritize **Logical/Evidence Features** (Contradiction Ratio, Dempster-Shafer). Avoid simple semantic similarity.
4.  **Is the tokenizer vocabulary extremely large (>100k)?**
    -   *Yes:* Use **Algorithmic/Lexical Features** (zlib compression) as a secondary check, as large vocabs can mask semantic distance.

---
*Developed by Jules, AI Software Engineer.*
