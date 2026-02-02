#!/usr/bin/env python3
"""
LEMON Full Pipeline - Download + Process (Raw + FOOOF)
Run this on RunPod: python3 pod_lemon_full.py
"""
import os
import sys
import subprocess

# Install dependencies first
print("Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mne", "fooof", "tqdm", "scipy", "pandas", "numpy"], check=True)

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch
from pathlib import Path
from tqdm import tqdm
import urllib.request
import re
import time
import json
import warnings
warnings.filterwarnings('ignore')

import mne
mne.set_log_level('ERROR')

from fooof import FOOOF

# Directories
DATA_DIR = Path('/workspace/lemon_data')
RESULTS_DIR = Path('/workspace/results')
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = 'https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID/'

CONFIG = {
    'theta_band': (4, 8),
    'alpha_band': (8, 13),
    'posterior_channels': ['O1', 'O2', 'Oz', 'P3', 'P4', 'Pz', 'P7', 'P8', 'PO3', 'PO4', 'POz'],
    'phi': 1.618034
}

# ========== DOWNLOAD FUNCTIONS ==========

def get_subject_list():
    print("Getting LEMON subject list from FTP...")
    with urllib.request.urlopen(BASE_URL, timeout=60) as response:
        html = response.read().decode()
    subjects = re.findall(r'href="(sub-\d+)/"', html)
    print(f"Found {len(subjects)} subjects")
    return sorted(subjects)

def download_subject(subj, max_retries=5):
    subj_dir = DATA_DIR / subj
    subj_dir.mkdir(exist_ok=True)
    
    for ext in ['.vhdr', '.vmrk', '.eeg']:
        filename = f'{subj}{ext}'
        url = f'{BASE_URL}{subj}/RSEEG/{filename}'
        local_path = subj_dir / filename
        
        if local_path.exists() and local_path.stat().st_size > 0:
            continue
        
        for attempt in range(max_retries):
            try:
                urllib.request.urlretrieve(url, local_path)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return False
    return True

def download_all():
    print("\n" + "="*60)
    print("STEP 1: DOWNLOADING LEMON DATASET")
    print("="*60)
    
    subjects = get_subject_list()
    success = 0
    
    for subj in tqdm(subjects, desc="Downloading"):
        if download_subject(subj):
            success += 1
    
    print(f"\nDownloaded: {success}/{len(subjects)} subjects")
    return subjects

# ========== PROCESSING FUNCTIONS ==========

def get_posterior_indices(ch_names):
    indices = []
    for i, ch in enumerate(ch_names):
        ch_clean = ch.upper().replace('.', '').replace('-', '')
        if any(t in ch_clean for t in CONFIG['posterior_channels']):
            indices.append(i)
    return indices

def compute_raw_centroids(freqs, psd):
    theta_mask = (freqs >= CONFIG['theta_band'][0]) & (freqs <= CONFIG['theta_band'][1])
    alpha_mask = (freqs >= CONFIG['alpha_band'][0]) & (freqs <= CONFIG['alpha_band'][1])
    
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
        theta_raw, alpha_raw = compute_raw_centroids(freqs, psd)
        res['theta_raw'] = theta_raw
        res['alpha_raw'] = alpha_raw
        if not np.isnan(theta_raw) and not np.isnan(alpha_raw):
            ratio = alpha_raw / theta_raw
            pci = np.log((abs(ratio - 2.0) + 0.1) / (abs(ratio - CONFIG['phi']) + 0.1))
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
            pci = np.log((abs(ratio - 2.0) + 0.1) / (abs(ratio - CONFIG['phi']) + 0.1))
            conv = 1 / (abs(alpha_fooof - theta_fooof) + 0.5)
            res['ratio_fooof'] = ratio
            res['pci_fooof'] = pci
            res['conv_fooof'] = conv
        
        return res
    except:
        return None

def process_all():
    print("\n" + "="*60)
    print("STEP 2: PROCESSING LEMON (Raw + FOOOF)")
    print("="*60)
    
    subj_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    print(f"Found {len(subj_dirs)} subjects to process")
    
    results = []
    for subj_dir in tqdm(subj_dirs, desc="Processing"):
        res = process_subject(subj_dir)
        if res:
            results.append(res)
        
        if len(results) % 50 == 0 and len(results) > 0:
            pd.DataFrame(results).to_csv(RESULTS_DIR / 'lemon_checkpoint.csv', index=False)
    
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'lemon_results.csv', index=False)
    print(f"\nSaved {len(df)} subjects")
    return df

def analyze_results(df):
    print("\n" + "="*60)
    print("STEP 3: ANALYSIS")
    print("="*60)
    
    # Raw analysis
    raw_valid = df.dropna(subset=['pci_raw', 'conv_raw'])
    if len(raw_valid) > 2:
        r, p = stats.pearsonr(raw_valid['pci_raw'], raw_valid['conv_raw'])
        rho, _ = stats.spearmanr(raw_valid['pci_raw'], raw_valid['conv_raw'])
        n = len(raw_valid)
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        ci_low, ci_high = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        phi_org = (raw_valid['pci_raw'] > 0).sum()
        
        print(f"\nRAW PSD (N={len(raw_valid)}):")
        print(f"  Pearson r = {r:.4f}, p = {p:.2e}")
        print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
        print(f"  Spearman rho = {rho:.4f}")
        print(f"  Phi-organized: {phi_org}/{len(raw_valid)} ({100*phi_org/len(raw_valid):.1f}%)")
        print(f"  Mean theta: {df['theta_raw'].mean():.2f} Hz")
        print(f"  Mean alpha: {df['alpha_raw'].mean():.2f} Hz")
        print(f"  Mean ratio: {df['ratio_raw'].mean():.3f}")
    
    # FOOOF analysis
    fooof_valid = df.dropna(subset=['pci_fooof', 'conv_fooof'])
    if len(fooof_valid) > 2:
        r_f, p_f = stats.pearsonr(fooof_valid['pci_fooof'], fooof_valid['conv_fooof'])
        rho_f, _ = stats.spearmanr(fooof_valid['pci_fooof'], fooof_valid['conv_fooof'])
        phi_org_f = (fooof_valid['pci_fooof'] > 0).sum()
        
        print(f"\nFOOOF (N={len(fooof_valid)}):")
        print(f"  Pearson r = {r_f:.4f}, p = {p_f:.2e}")
        print(f"  Spearman rho = {rho_f:.4f}")
        print(f"  Phi-organized: {phi_org_f}/{len(fooof_valid)} ({100*phi_org_f/len(fooof_valid):.1f}%)")
        print(f"  Mean theta: {df['theta_fooof'].mean():.2f} Hz")
        print(f"  Mean alpha: {df['alpha_fooof'].mean():.2f} Hz")
        print(f"  Mean ratio: {df['ratio_fooof'].mean():.3f}")
        print(f"  Mean exponent: {df['exponent'].mean():.3f}")
    
    # Auxiliary analyses
    print("\n" + "="*60)
    print("AUXILIARY ANALYSES")
    print("="*60)
    
    valid = df.dropna(subset=['theta_raw', 'alpha_raw', 'theta_fooof', 'alpha_fooof', 'exponent', 'pci_raw', 'conv_raw'])
    if len(valid) > 2:
        theta_r, _ = stats.pearsonr(valid['theta_raw'], valid['theta_fooof'])
        alpha_r, _ = stats.pearsonr(valid['alpha_raw'], valid['alpha_fooof'])
        print(f"\nTheta raw-FOOOF: r = {theta_r:.4f}")
        print(f"Alpha raw-FOOOF: r = {alpha_r:.4f}")
        
        exp_pci_r, exp_pci_p = stats.pearsonr(valid['exponent'], valid['pci_raw'])
        exp_conv_r, exp_conv_p = stats.pearsonr(valid['exponent'], valid['conv_raw'])
        print(f"\n1/f exponent vs PCI_raw: r = {exp_pci_r:.4f}, p = {exp_pci_p:.2e}")
        print(f"1/f exponent vs conv_raw: r = {exp_conv_r:.4f}, p = {exp_conv_p:.2e}")
        
        # Partial correlation
        pci_resid = valid['pci_raw'] - np.polyval(np.polyfit(valid['exponent'], valid['pci_raw'], 1), valid['exponent'])
        conv_resid = valid['conv_raw'] - np.polyval(np.polyfit(valid['exponent'], valid['conv_raw'], 1), valid['exponent'])
        partial_r, partial_p = stats.pearsonr(pci_resid, conv_resid)
        orig_r, _ = stats.pearsonr(valid['pci_raw'], valid['conv_raw'])
        print(f"\nPartial corr (controlling 1/f): r = {partial_r:.4f}, p = {partial_p:.2e}")
        print(f"Original corr: r = {orig_r:.4f}")
    
    # Save summary
    summary = {
        'dataset': 'LEMON',
        'n_total': len(df),
        'raw': {
            'n_valid': len(raw_valid) if 'raw_valid' in dir() else 0,
            'pearson_r': round(r, 4) if 'r' in dir() else None,
            'pearson_p': f'{p:.2e}' if 'p' in dir() else None,
            'phi_organized_pct': round(100*phi_org/len(raw_valid), 1) if 'phi_org' in dir() else None,
            'mean_ratio': round(df['ratio_raw'].mean(), 3) if 'ratio_raw' in df.columns else None
        },
        'fooof': {
            'n_valid': len(fooof_valid) if 'fooof_valid' in dir() else 0,
            'pearson_r': round(r_f, 4) if 'r_f' in dir() else None,
            'mean_exponent': round(df['exponent'].mean(), 3) if 'exponent' in df.columns else None
        }
    }
    
    with open(RESULTS_DIR / 'lemon_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_DIR}")

if __name__ == '__main__':
    download_all()
    df = process_all()
    analyze_results(df)
    print("\n" + "="*60)
    print("DONE! Results in /workspace/results/")
    print("="*60)
