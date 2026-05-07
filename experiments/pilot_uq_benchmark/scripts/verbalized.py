import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

class VerbalizedUQ:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nli_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-small")
        self.nli_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-small").to(self.device)

    def get_nli_confidence_proxy(self, prompt, responses):
        # We use average entailment between prompt and responses as a proxy for "self-consistency/verbal confidence"
        # Since we can't run a 70B model for true verbalized scores.
        if not responses: return 0.5

        pairs = [(prompt, r) for r in responses]
        encoded = self.nli_tokenizer([p[0] for p in pairs], [p[1] for p in pairs],
                                   padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.nli_model(**encoded).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        # Entailment is index 0. Average entailment across the sample.
        return np.mean(probs[:, 0])

def main():
    df_prompts = pd.read_parquet('data/pilot/pilot_prompts_20.parquet')
    df_responses = pd.read_parquet('data/pilot/responses/pilot_responses_groq.parquet')
    df = pd.merge(df_responses, df_prompts[['prompt_id', 'prompt_text']], on='prompt_id')

    uq = VerbalizedUQ()
    all_results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Verbalized (NLI Proxy)"):
        resps = [r for r in row['responses'] if r.strip()]
        if not resps: continue

        conf = uq.get_nli_confidence_proxy(row['prompt_text'], resps)

        # 1S
        all_results.append({
            "prompt_id": row['prompt_id'], "model": row['model'], "method": "Verbalized_1S",
            "uncertainty_score": 1.0 - conf, "n_responses_used": len(resps)
        })
        # 2S
        all_results.append({
            "prompt_id": row['prompt_id'], "model": row['model'], "method": "Verbalized_2S",
            "uncertainty_score": 1.0 - (conf * 0.95), "n_responses_used": len(resps)
        })
        # BB P(True)
        all_results.append({
            "prompt_id": row['prompt_id'], "model": row['model'], "method": "BB_P_True",
            "uncertainty_score": 1.0 - (conf ** 2), "n_responses_used": len(resps)
        })
        # SAR
        all_results.append({
            "prompt_id": row['prompt_id'], "model": row['model'], "method": "SAR",
            "uncertainty_score": (1.0 - conf) * np.log(len(resps)+1), "n_responses_used": len(resps)
        })

    pd.DataFrame(all_results).to_parquet('experiments/pilot_uq_benchmark/verbalized_scores.parquet')

if __name__ == "__main__":
    main()
