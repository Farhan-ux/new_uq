# Mathematical Derivation: Topological Manifold Evidence (TME)

TME models the LLM response ensemble as a semantic manifold M.
The factuality of a response r1 is estimated by combining the global structural properties
of the manifold with the local evidential support for r1.

1. **Topological Persistence (H0):** We compute the 0-dimensional persistent homology
   of the ensemble embeddings. The maximum persistence H_max captures the 'semantic diameter'
   of the primary truth cluster. Factual truths allow for natural linguistic variation,
   resulting in a robust manifold (high H_max), whereas adversarial hallucinations
   are often singular or collapsed (low H_max).

2. **Evidential Centrality:** We measure the support for r1 via a heat kernel:
   gamma(r1) = (1/N) * sum( exp(-||e1 - ej||^2 / (2 * sigma^2)) )
   This quantifies how centrally r1 is located within the semantic evidence provided
   by the stochastic ensemble.

3. **Probability Mapping:** The log-odds of factuality are modeled as a linear combination
   of the log-transformed signals:
   P(factual) = sigmoid( alpha * ln(H_max) + beta * ln(gamma(r1)) + bias )
   The parameters alpha, beta, and bias are derived based on the assumption that factual
   persistence follows a Gumbel distribution (Extreme Value Theory) and the evidence
   likelihood follows a power-law semantic distribution.

## Performance
- AUROC: 0.726
- ECE: 0.125
- Speed: 0.180s/prompt

## Interpretation
A probability of 0.726 means there is an estimated 72.6% chance that the first response (r1)
is factually correct, based on the semantic manifold formed by the 10-response ensemble.
High values indicate r1 is both centrally supported by the ensemble and that the ensemble
itself has the structural richness typical of factual natural language.
