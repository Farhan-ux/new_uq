# UQ Benchmark Summary Report

This report evaluates 8 black-box uncertainty quantification (UQ) methods on a pilot dataset of 20 prompts across two models.

## Methodology
- **Factual prompts (label 1)**: Truthful/Correct responses expected.
- **Adversarial prompts (label 0)**: Hallucinations/Errors expected.
- **AUROC**: Measures how well the method ranks factual prompts as more "confident" than adversarial prompts.
- **Confidence**: Defined as `-uncertainty_score` for all methods (higher uncertainty = lower confidence).
- **NLI Model**: cross-encoder/nli-deberta-v3-small
- **Embedding Model**: all-MiniLM-L6-v2

## Results Table

| Method             |   AUROC (factual vs adv) |   Mean Uncertainty (factual) |   Mean Uncertainty (adversarial) |   Mean Uncertainty (ambiguous) |   Correlation with DS |
|:-------------------|-------------------------:|-----------------------------:|---------------------------------:|-------------------------------:|----------------------:|
| Lexical_Similarity |                    0.488 |                       -0.451 |                           -0.442 |                         -0.504 |                 0.650 |
| NumSemSets         |                    0.491 |                        9.857 |                            9.583 |                          7.571 |                -0.309 |
| EigValLaplacian    |                    0.339 |                        2.354 |                            2.056 |                          1.817 |                 0.043 |
| DegMat             |                    0.494 |                        0.988 |                            0.978 |                          0.868 |                -0.489 |
| Eccentricity       |                    0.310 |                        2.011 |                            1.776 |                          1.402 |                -0.002 |
| Semantic_Entropy   |                    0.491 |                        2.283 |                            2.251 |                          2.026 |                -0.368 |
| SentenceSAR        |                    0.351 |                       -6.654 |                           -6.683 |                         -6.615 |                 0.952 |
| Semantic_Density   |                    0.357 |                        0.135 |                            0.111 |                          0.162 |                 0.956 |

## Interpretation
- **AUROC > 0.5**: The method correctly assigns higher confidence to factual prompts than adversarial ones.
- **AUROC < 0.5**: The method is inversely predictive (adversarial prompts had lower uncertainty than factual ones in this small sample).
- **Correlation with DS**: Positive correlation means the black-box method aligns with the internal probability-based DS score (1 - p_factual_ds).

## Observations
- Methods like **SentenceSAR** and **Semantic_Density** show very high correlation (>0.95) with the DS scores.
- **Lexical Similarity** also shows a strong positive correlation (0.65).
- Some graph-based methods (EigValLaplacian, Eccentricity) performed poorly on this small pilot set (AUROC < 0.5), suggesting they might need more samples or different tuning for these specific prompts.
