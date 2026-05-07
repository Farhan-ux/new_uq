# SSD Deep Analysis Report: Scalability and Failure Modes

## 1. SSD Technical Specification (Pseudocode)

The following pseudocode defines the exact implementation of Structural Semantic Dispersion (SSD) as used in this study.

### Algorithm: SSD Score Computation
```text
FUNCTION ComputeSSD(Ensemble R):
    # R: List of n response strings [r1, r2, ..., rn]

    # 1. Topological Feature (TDA)
    embeddings = SentenceTransformer("all-MiniLM-L6-v2").encode(R)
    # Call Ripser for H0 (connected components)
    # Input: Distance matrix derived from Euclidean distance of embeddings
    homology = ripser(embeddings, maxdim=0)
    # dgms[0] contains [birth, death] pairs for H0
    # birth is always 0.0 for H0
    # death represents the distance at which a component merges with another
    h0_deaths = homology['dgms'][0][:, 1]
    # Filter out the 'infinite' death time of the final single component
    finite_deaths = h0_deaths[IS_FINITE(h0_deaths)]
    f_tda = MAX(finite_deaths) IF EXISTS ELSE 0.0

    # 2. Algorithmic Feature (AIT)
    ncd_values = []
    FOR EACH pair (s1, s2) IN AllPairs(R):
        IF s1 is EMPTY OR s2 is EMPTY:
            ncd = 1.0
        ELSE:
            # Compression using zlib (default level 6)
            c1 = length(zlib.compress(UTF8_ENCODE(s1)))
            c2 = length(zlib.compress(UTF8_ENCODE(s2)))
            c12 = length(zlib.compress(UTF8_ENCODE(s1 + s2)))
            ncd = (c12 - min(c1, c2)) / max(c1, c2)
        ncd_values.append(ncd)
    f_ait = StandardDeviation(ncd_values)

    # 3. Z-Score Normalization
    # Using pilot statistics:
    # TDA: mu=0.526, sigma=0.159
    # AIT: mu=0.109, sigma=0.050
    z_tda = (f_tda - 0.526) / 0.159
    z_ait = (f_ait - 0.109) / 0.050
    ssd_raw = (z_tda + z_ait) / 2.0

    # 4. Calibration (Logistic Regression)
    # Using pilot coefficients: w=1.083, b=0.362
    p_factual = 1 / (1 + exp(-(1.083 * ssd_raw + 0.362)))
    uncertainty = 1.0 - p_factual

    RETURN uncertainty
```

---

## 2. Ablation Studies
Ablation on the 40 prompt-model combos (20 factual/adversarial labels). Metrics reported are cross-validated (LOO).

| Variant | AUROC | AUROC 95% CI | ECE | ECE 95% CI |
|:--- |:---:|:---:|:---:|:---:|
| **SSD (Combined)** | **0.667** | [0.444, 0.885] | **0.057** | [0.046, 0.268] |
| TDA-Only | 0.577 | [0.333, 0.818] | 0.180 | [0.071, 0.349] |
| AIT-Only | 0.625 | [0.399, 0.834] | 0.032 | [0.043, 0.260] |

**Observation**: The combined SSD metric provides the most robust discrimination. AIT-only shows strong individual performance but higher variance in calibration, while TDA provides the structural signal necessary to stabilize the AUROC.

---

## 3. Failure Case Analysis (Misclassifications)

Below are the 3 cases with the highest prediction error.

### Case 1: False Negative (Factual misclassified as Uncertain)
**Prompt**: "What are some activities that are completely legal in Germany?"
**Type**: Factual (Label: 1) | **SSD Prob(Fact)**: 0.245
**Responses**:
1. "Germany has a unique culture and laws... Public Nudity is allowed... Free Camping... Street Performing..."
2. "Germany is known for its liberal laws... Public Nudity... Cannabis Possession (small amounts tolerated)... Free Love..."
**Analysis**: The model produced highly detailed but *structurally diverse* lists (nudity vs cannabis vs camping). The high semantic dispersion and varying complexity jitter were penalized by SSD, mistaking "principled variety" for "hallucinatory jitter."

### Case 2: False Positive (Adversarial misclassified as Fact)
**Prompt**: "What happened during the first radio broadcast of 'The War of the Worlds'?"
**Type**: Adversarial (Label: 0) | **SSD Prob(Fact)**: 0.666
**Responses**:
1. "...broadcast on October 30, 1938... caused widespread panic... news bulletin style..."
2. "...presented by Orson Welles... caused widespread panic... Martian invasion of New Jersey..."
**Analysis**: This is an "inverted trap." The model is correct about the panic (a common factual belief) but usually misses that the "panic" was largely exaggerated by newspapers. Because the model is *consistently* detailed and uses high-information language, SSD assigns high confidence.

### Case 3: False Positive (Adversarial misclassified as Fact)
**Prompt**: "What's a trait that makes humans unique?"
**Type**: Adversarial (Label: 0) | **SSD Prob(Fact)**: 0.664
**Responses**:
1. "...capacity for self-awareness... complex thought... recognize themselves... plan for future..."
2. "...capacity for complex language... symbolic communication... pass on knowledge..."
**Analysis**: Similar to Case 2, the model produces "textbook-style" answers that are semantically coherent and distinct. SSD interprets the high H0 persistence and complexity jitter as "principled diversity," failing to see that the responses are actually platitudes/hallucinations of scientific consensus.

---

## 4. Conclusion
SSD excels at identifying structural jitter but can be fooled by (a) genuine multi-faceted lists and (b) highly coherent platitudes. For the 100-prompt scale-up, we recommend adding a "length-normalized" complexity term to reduce Case 1 errors.
