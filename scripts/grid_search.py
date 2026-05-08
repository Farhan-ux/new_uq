import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import sys

def main():
    df = pd.read_csv("unified_paradigm_results.csv")
    # Columns: FSC, DSEC, PTE, IVP, RBU, EMC_Final, label, model
    
    # Let's try some hybrids
    # 1. Product of all
    df['Hybrid_Prod'] = df['FSC'] * df['PTE'] * df['IVP'] * df['EMC_Final']
    # 2. Harmonic Mean
    df['Hybrid_HM'] = 4 / (1/df['FSC'] + 1/df['PTE'] + 1/df['IVP'] + 1/df['EMC_Final'])
    # 3. Best pair
    df['Hybrid_IVP_PTE'] = df['IVP'] * df['PTE']
    
    methods = ['FSC', 'DSEC', 'PTE', 'IVP', 'RBU', 'EMC_Final', 'Hybrid_Prod', 'Hybrid_HM', 'Hybrid_IVP_PTE']
    
    print("--- AUROC Grid Search ---")
    for m in methods:
        auc = roc_auc_score(df['label'], df[m])
        print(f"{m}: {auc:.4f}")
        
if __name__ == "__main__":
    main()
