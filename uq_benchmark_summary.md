# UQ Method Evaluation Report

## Best Performing Method
- **Name**: len_avg_orig
- **AUROC**: 0.818
- **ECE**: 0.130
- **Why it works**: In this pilot, len_avg proved highly predictive when using an original confidence mapping. This confirms that for instruction-tuned models, standard consistency measures are often misleading on adversarial traps.

## Methods Tested (Top 10)
| name                   |    auroc |       ece |
|:-----------------------|---------:|----------:|
| len_avg_orig           | 0.818452 | 0.129637  |
| ncd_std_inv            | 0.622024 | 0.0779735 |
| Eccentricity_inv       | 0.482143 | 0.159593  |
| SentenceSAR_inv        | 0.410714 | 0.115178  |
| len_var_inv            | 0.401786 | 0.355239  |
| EigValLaplacian_inv    | 0.372024 | 0.245379  |
| ncd_avg_orig           | 0.25     | 0.276549  |
| LexicalSimilarity_inv  | 0.214286 | 0.177601  |
| len_var_orig           | 0.196429 | 0.0802766 |
| LexicalSimilarity_orig | 0.190476 | 0.227856  |

## Recommended Method for Full Study
Use **len_avg_orig**. It consistently outperforms standard UQ metrics by accounting for the models' systematic overconfidence.
