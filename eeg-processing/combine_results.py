"""
================================================================================
COMBINE RESULTS: Merge PhysioNet + LEMON results
================================================================================
Run locally after downloading results from both pods
================================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

def combine_and_analyze():
    print("="*60)
    print("COMBINING RESULTS: PhysioNet + LEMON")
    print("="*60)
    
    # Load results
    df_physio = pd.read_csv('results/physionet_results.csv')
    df_lemon = pd.read_csv('results/lemon_results.csv')
    
    print(f"\nPhysioNet: N={len(df_physio)}")
    print(f"LEMON: N={len(df_lemon)}")
    
    # Combine
    df_all = pd.concat([df_physio, df_lemon], ignore_index=True)
    df_all.to_csv('results/combined_results.csv', index=False)
    print(f"Combined: N={len(df_all)}")
    
    # Run analysis for each method
    for method in ['raw', 'fooof']:
        pci_col = f'pci_posterior_{method}'
        conv_col = f'conv_bound_posterior_{method}'
        
        if pci_col not in df_all.columns:
            continue
        
        valid = df_all[[pci_col, conv_col, 'dataset']].dropna()
        
        print(f"\n{'='*60}")
        print(f"ANALYSIS: {method.upper()} METHOD")
        print(f"{'='*60}")
        print(f"Valid N: {len(valid)}")
        
        # Main correlation
        r, p = stats.pearsonr(valid[pci_col], valid[conv_col])
        rho, _ = stats.spearmanr(valid[pci_col], valid[conv_col])
        
        n = len(valid)
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        ci_low, ci_high = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        
        print(f"\n1. MAIN CORRELATION")
        print(f"   Pearson r = {r:.3f}, p = {p:.2e}")
        print(f"   95% CI [{ci_low:.3f}, {ci_high:.3f}]")
        print(f"   Spearman ρ = {rho:.3f}")
        
        # Phi-organized
        phi_org = (valid[pci_col] > 0).sum()
        phi_pct = 100 * phi_org / len(valid)
        print(f"\n2. PHI-ORGANIZED: {phi_org}/{len(valid)} ({phi_pct:.1f}%)")
        
        # Frontal validation
        pci_front = f'pci_frontal_{method}'
        conv_front = f'conv_bound_frontal_{method}'
        if pci_front in df_all.columns:
            valid_front = df_all[[pci_front, conv_front]].dropna()
            if len(valid_front) > 10:
                r_front, _ = stats.pearsonr(valid_front[pci_front], valid_front[conv_front])
                print(f"\n3. FRONTAL VALIDATION")
                print(f"   Frontal r = {r_front:.3f}")
                print(f"   Posterior r = {r:.3f}")
                print(f"   Δr = {r_front - r:.3f}")
        
        # By dataset
        print(f"\n4. BY DATASET")
        for ds in ['PhysioNet', 'LEMON']:
            ds_data = valid[valid['dataset'] == ds]
            if len(ds_data) > 10:
                r_ds, p_ds = stats.pearsonr(ds_data[pci_col], ds_data[conv_col])
                print(f"   {ds}: N={len(ds_data)}, r={r_ds:.3f}")
        
        # Null model
        print(f"\n5. NULL MODEL")
        theta_col = f'theta_posterior_{method}'
        alpha_col = f'alpha_posterior_{method}'
        theta_vals = df_all[theta_col].dropna().values
        alpha_vals = df_all[alpha_col].dropna().values
        
        if len(theta_vals) > 10:
            np.random.seed(42)
            null_rs = []
            for _ in range(1000):
                theta_perm = np.random.permutation(theta_vals)
                alpha_perm = np.random.permutation(alpha_vals)
                ratio_perm = alpha_perm / theta_perm
                pci_perm = np.log((np.abs(ratio_perm - 2.0) + 0.1) / (np.abs(ratio_perm - 1.618034) + 0.1))
                conv_perm = 1 / (np.abs(alpha_perm - theta_perm) + 0.5)
                r_null, _ = stats.pearsonr(pci_perm, conv_perm)
                null_rs.append(r_null)
            
            null_mean, null_sd = np.mean(null_rs), np.std(null_rs)
            z_score = (r - null_mean) / null_sd
            print(f"   Null mean = {null_mean:.3f}, SD = {null_sd:.3f}")
            print(f"   Observed r = {r:.3f}, Z = {z_score:.2f}")
    
    # Summary table for manuscript
    print(f"\n{'='*60}")
    print("SUMMARY FOR MANUSCRIPT")
    print(f"{'='*60}")
    print(f"""
MULTI-DATASET RESULTS (N={len(df_all)}):
─────────────────────────────────────────
PhysioNet: N={len(df_physio)}
LEMON: N={len(df_lemon)}

Copy the statistics above into your manuscript.
Results saved to: results/combined_results.csv
""")

if __name__ == "__main__":
    combine_and_analyze()
