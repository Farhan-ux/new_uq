# Mathematical Derivation: Topological Manifold Evidence (TME)

TME models the LLM response ensemble as a semantic manifold $M \subset \mathbb{R}^d$.
The factuality of a response $r_1$ is estimated by combining the global structural properties
of the manifold with the local evidential support for $r_1$.

1. **Topological Persistence ($H_0$):** We compute the 0-dimensional persistent homology
   of the ensemble embeddings. The maximum persistence $H_{max}$ captures the 'semantic diameter'
   of the primary truth cluster. Factual truths allow for natural linguistic variation,
   resulting in a robust manifold ($\text{high } H_{max}$), whereas adversarial hallucinations
   are often singular or collapsed ($\text{low } H_{max}$).

2. **Evidential Centrality:** We measure the support for $r_1$ via a heat kernel:
   $$\gamma(r_1) = \frac{1}{N} \sum_{j=1}^N \exp\left(-\frac{\|e_1 - e_j\|^2}{2\sigma^2}\right)$$
   This quantifies how centrally $r_1$ is located within the semantic evidence provided
   by the stochastic ensemble.

3. **Probability Mapping:** The log-odds of factuality are modeled as a linear combination
   of the log-transformed signals:
   $$P(\text{factual}) = \sigma( \alpha \ln(H_{max}) + \beta \ln(\gamma(r_1)) + \text{bias} )$$
   The parameters $\alpha, \beta, \text{bias}$ are derived based on the assumption that factual
   persistence follows a Gumbel distribution (Extreme Value Theory).

## Performance
- AUROC: 0.726
- ECE: 0.125
- Speed: 0.168s/prompt
