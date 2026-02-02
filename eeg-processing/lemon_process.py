#!/usr/bin/env python3
"""LEMON Dataset Processing - Raw + FOOOF"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch
from pathlib import Path
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

import mne
mne.set_log_level('ERROR')

from fooof import FOOOF

DATA_DIR = Path('/workspace/lemon_data')
RESULTS_DIR = Path('/workspace/results')
RESULTS_DIR.mkdir(exist_ok=True)

CONFIG = {
    'theta_band': (4, 8),
    'alpha_band': (8, 13),
    'posterior_channels': ['O1', 'O2', 'Oz', 'P3', 'P4', 'Pz', 'P7', 'P8', 'PO3', 'PO4', 'POz'],
    'phi': 1.618034
}

def get_posterior_indices(ch_names):
    indices = []
    for i, ch in enumerate(ch_names):
        ch_clean = ch.upper().replace('.', '').replace('-', '')
        if any(t in ch_clean for t in CONFIG['posterior_channels']):
            indices.append(i)
    return indices

def compute_raw_centroids(freqs, psd, theta_band, alpha_band):
    theta_mask = (freqs >= theta_band[0]) & (freqs <= theta_band[1])
    alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
    
    theta = np.average(freqs[theta_mask], weights=psd[theta_mask]) if psd[theta_mask].sum() > 0 else np.nan
    alpha = np.average(freqs[alpha_mask], weights=psd[alpha_mask]) if psd[alpha_mask].sum() > 0 else np.nan
    return theta, alpha

def compute_fooof_centroids(freqs, psd):
    try:
        fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.05, verbose=False)
        fm.fit(freqs, psd, [2, 40])
        
        if not fm.has_model:
            return np.nan, np.nan, np.nan
        
        periodic = fm.fooofed_spectrum_ - fm._ap_fit
        periodic = np.maximum(periodic, 0)
        
        theta_mask = (fm.freqs >= 4) & (fm.freqs <= 8)
        alpha_mask = (fm.freqs >= 8) & (fm.freqs <= 13)
        
        theta = np.average(fm.freqs[theta_mask], weights=periodic[theta_mask]) if periodic[theta_mask].sum() > 0 else np.nan
        alpha = np.average(fm.freqs[alpha_mask], weights=periodic[alpha_mask]) if periodic[alpha_mask].sum() > 0 else np.nan
        exponent = fm.aperiodic_params_[1]
        
        return theta, alpha, exponent
    except:
        return np.nan, np.nan, np.nan

def process_subject(subj_dir):
    """Process one subject - return dict with raw and FOOOF results"""
    vhdr_files = list(subj_dir.glob('*.vhdr'))
    if not vhdr_files:
        return None
    
    try:
        raw = mne.io.read_raw_brainvision(str(vhdr_files[0]), preload=True, verbose=False)
        raw.filter(1, 45, verbose=False)
        
        ch_idx = get_posterior_indices(raw.ch_names)
        if len(ch_idx) < 2:
            return None
        
        data = raw.get_data()[ch_idx].mean(axis=0)
        fs = raw.info['sfreq']
        
        freqs, psd = welch(data, fs, nperseg=int(fs*2))
        freq_mask = (freqs >= 1) & (freqs <= 45)
        freqs, psd = freqs[freq_mask], psd[freq_mask]
        
        res = {'subject': subj_dir.name, 'dataset': 'LEMON'}
        
        # Raw PSD
        theta_raw, alpha_raw = compute_raw_centroids(freqs, psd, CONFIG['theta_band'], CONFIG['alpha_band'])
        res['theta_raw'] = theta_raw
        res['alpha_raw'] = alpha_raw
        if not np.isnan(theta_raw) and not np.isnan(alpha_raw):
            ratio = alpha_raw / theta_raw
            pci = np.log((abs(ratio - 2.0) + 0.1) / (abs(ratio - 1.618034) + 0.1))
            conv = 1 / (abs(alpha_raw - theta_raw) + 0.5)
            res['ratio_raw'] = ratio
            res['pci_raw'] = pci
            res['conv_raw'] = conv
        
        # FOOOF
        theta_fooof, alpha_fooof, exponent = compute_fooof_centroids(freqs, psd)
        res['theta_fooof'] = theta_fooof
        res['alpha_fooof'] = alpha_fooof
        res['exponent'] = exponent
        if not np.isnan(theta_fooof) and not np.isnan(alpha_fooof):
            ratio = alpha_fooof / theta_fooof
            pci = np.log((abs(ratio - 2.0) + 0.1) / (abs(ratio - 1.618034) + 0.1))
            conv = 1 / (abs(alpha_fooof - theta_fooof) + 0.5)
            res['ratio_fooof'] = ratio
            res['pci_fooof'] = pci
            res['conv_fooof'] = conv
        
        return res
    except Exception as e:
        return None

def main():
    print("="*60)
    print("LEMON Processing (Raw + FOOOF)")
    print("="*60)
    
    subj_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    print(f"Found {len(subj_dirs)} subjects")
    
    results = []
    for subj_dir in tqdm(subj_dirs, desc="Processing"):
        res = process_subject(subj_dir)
        if res:
            results.append(res)
        
        # Checkpoint every 50 subjects
        if len(results) % 50 == 0 and len(results) > 0:
            pd.DataFrame(results).to_csv(RESULTS_DIR / 'lemon_checkpoint.csv', index=False)
    
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'lemon_results.csv', index=False)
    print(f"\nSaved {len(df)} subjects")
    
    # Analysis
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    # Raw analysis
    raw_valid = df.dropna(subset=['pci_raw', 'conv_raw'])
    if len(raw_valid) > 2:
        r, p = stats.pearsonr(raw_valid['pci_raw'], raw_valid['conv_raw'])
        rho, _ = stats.spearmanr(raw_valid['pci_raw'], raw_valid['conv_raw'])
        phi_org = (raw_valid['pci_raw'] > 0).sum()
        
        print(f"\nRAW PSD (N={len(raw_valid)}):")
        print(f"  Pearson r = {r:.4f}, p = {p:.2e}")
        print(f"  Spearman rho = {rho:.4f}")
        print(f"  Phi-organized: {phi_org}/{len(raw_valid)} ({100*phi_org/len(raw_valid):.1f}%)")
        print(f"  Mean ratio: {df['ratio_raw'].mean():.3f}")
    
    # FOOOF analysis
    fooof_valid = df.dropna(subset=['pci_fooof', 'conv_fooof'])
    if len(fooof_valid) > 2:
        r, p = stats.pearsonr(fooof_valid['pci_fooof'], fooof_valid['conv_fooof'])
        rho, _ = stats.spearmanr(fooof_valid['pci_fooof'], fooof_valid['conv_fooof'])
        phi_org = (fooof_valid['pci_fooof'] > 0).sum()
        
        print(f"\nFOOOF (N={len(fooof_valid)}):")
        print(f"  Pearson r = {r:.4f}, p = {p:.2e}")
        print(f"  Spearman rho = {rho:.4f}")
        print(f"  Phi-organized: {phi_org}/{len(fooof_valid)} ({100*phi_org/len(fooof_valid):.1f}%)")
        print(f"  Mean ratio: {df['ratio_fooof'].mean():.3f}")
        print(f"  Mean exponent: {df['exponent'].mean():.3f}")
    
    # Auxiliary analyses
    print("\n" + "="*60)
    print("AUXILIARY ANALYSES")
    print("="*60)
    
    valid = df.dropna(subset=['theta_raw', 'alpha_raw', 'theta_fooof', 'alpha_fooof', 'exponent', 'pci_raw', 'conv_raw'])
    if len(valid) > 2:
        # Raw vs FOOOF correlations
        theta_r, _ = stats.pearsonr(valid['theta_raw'], valid['theta_fooof'])
        alpha_r, _ = stats.pearsonr(valid['alpha_raw'], valid['alpha_fooof'])
        print(f"\nTheta raw-FOOOF: r = {theta_r:.4f}")
        print(f"Alpha raw-FOOOF: r = {alpha_r:.4f}")
        
        # 1/f exponent relationships
        exp_pci_r, exp_pci_p = stats.pearsonr(valid['exponent'], valid['pci_raw'])
        exp_conv_r, exp_conv_p = stats.pearsonr(valid['exponent'], valid['conv_raw'])
        print(f"\n1/f exponent vs PCI_raw: r = {exp_pci_r:.4f}, p = {exp_pci_p:.2e}")
        print(f"1/f exponent vs conv_raw: r = {exp_conv_r:.4f}, p = {exp_conv_p:.2e}")
        
        # Partial correlation
        pci_resid = valid['pci_raw'] - np.polyval(np.polyfit(valid['exponent'], valid['pci_raw'], 1), valid['exponent'])
        conv_resid = valid['conv_raw'] - np.polyval(np.polyfit(valid['exponent'], valid['conv_raw'], 1), valid['exponent'])
        partial_r, partial_p = stats.pearsonr(pci_resid, conv_resid)
        orig_r, _ = stats.pearsonr(valid['pci_raw'], valid['conv_raw'])
        print(f"\nPartial corr (controlling 1/f): r = {partial_r:.4f}")
        print(f"Original corr: r = {orig_r:.4f}")
    
    # Save summary
    summary = {
        'dataset': 'LEMON',
        'n_subjects': len(df),
        'raw': {
            'n_valid': len(raw_valid) if 'raw_valid' in dir() else 0,
            'pearson_r': round(r, 4) if 'raw_valid' in dir() and len(raw_valid) > 2 else None
        }
    }
    with open(RESULTS_DIR / 'lemon_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == '__main__':
    main()
