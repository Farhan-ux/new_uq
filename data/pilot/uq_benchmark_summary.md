# Pilot UQ Benchmark Summary

| Method                         |   AUROC |   Mean Unc (Fact) |   Mean Unc (Adv) |   Mean Unc (Ambig) |   DS Corr |
|:-------------------------------|--------:|------------------:|-----------------:|-------------------:|----------:|
| Structural_Semantic_Dispersion |   0.768 |             0.388 |            0.547 |              0.376 |    -0.744 |
| Eccentricity                   |   0.595 |             0.994 |            0.992 |              0.9   |     0.126 |
| Semantic_Entropy               |   0.586 |             2.223 |            2.194 |              1.449 |    -0.365 |
| NumSemSets                     |   0.586 |             9.429 |            9.333 |              6.071 |    -0.362 |
| DegMat                         |   0.562 |            -0.207 |           -0.308 |             -1.686 |    -0.354 |
| Lexical_Similarity             |   0.536 |            -0.38  |           -0.355 |             -0.417 |     0.685 |
| BB_P_True                      |   0.536 |             0.976 |            0.999 |              0.88  |    -0.092 |
| Verbalized_2S                  |   0.53  |             0.956 |            0.987 |              0.782 |    -0.091 |
| SAR                            |   0.53  |             2.286 |            2.366 |              1.849 |    -0.091 |
| Verbalized_1S                  |   0.53  |             0.953 |            0.987 |              0.771 |    -0.091 |
| EigValLaplacian                |   0.438 |             2.071 |            3     |             16.786 |     0.355 |
| Semantic_Density               |   0.357 |            -0.878 |           -0.9   |             -0.854 |     0.972 |

*Note: DS Corr is Spearman correlation with inverted DS scores (1-p_factual_ds).*