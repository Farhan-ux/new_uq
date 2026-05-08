import sys
import os
# Ensure we can import from the scripts directory
sys.path.append(os.path.dirname(__file__))

from emc_estimator import EvidentialManifoldConsensus

def main():
    """
    Final Submission Script for the Probability Estimator Task.
    Uses the Evidential Manifold Consensus (EMC) method.
    """
    # 1. Initialize the estimator (strictly black-box)
    estimator = EvidentialManifoldConsensus()
    
    # 2. Example Input (10 responses to a prompt)
    # In practice, this would be passed from an external harness
    responses = [
        "The primary color of a clear daytime sky is blue.",
        "A clear sky appears blue during the day.",
        "Sky is blue.",
        "It's blue.",
        "Blue.",
        "The sky is blue because of sunlight scattering.",
        "Daytime sky color: blue.",
        "The sky is green.", # Deliberate hallucination in ensemble
        "Blue.",
        "A clear blue sky."
    ]
    
    # 3. Predict factuality probability for the first response
    p = estimator.estimate_factuality(responses)
    
    print(f"--- Factuality Estimate ---")
    print(f"Primary Response: {responses[0]}")
    print(f"Estimated Probability of Correctness: {p:.2%}")
    print(f"Interpretation: {p:.2%} means based on ensemble manifold and Bayesian consensus, ")
    print(f"there is a {p:.2%} estimated likelihood of factual correctness.")

if __name__ == "__main__":
    main()
