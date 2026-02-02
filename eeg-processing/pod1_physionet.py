"""
================================================================================
POD 1: PhysioNet EEGMMIDB Processing (N=109)
================================================================================
Run on RunPod with: 4 vCPU, 16GB RAM, 20GB storage
Estimated time: ~30 min
================================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch
from pathlib import Path
from tqdm import tqdm
import urllib.request
import warnings
warnings.filterwarnings('ignore')

import mne
mne.set_log_level('ERROR')

try:
    from specparam import SpectralModel
    HAS_FOOOF = True
    print("✓ FOOOF available")
except:
    HAS_FOOOF = False
    print("✗ FOOOF not available, using raw only")

CONFIG = {
    'theta_band': (4, 8),
    'alpha_band': (8, 13),
    'fooof_settings': {
        'peak_width_limits': [1, 8],
        'max_n_peaks': 6,
        'min_peak_height': 0.1,
        'aperiodic_mode': 'fixed'
    },
    'posterior_channels': ['O1', 'O2', 'Oz', 'P3', 'P4', 'Pz', 'P7', 'P8', 'PO3', 'PO4'],
    'frontal_channels': ['Fz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'AFz'],
    'phi': 1.618034,
    'epsilon': 0.1
}

def download_physionet(save_dir='data/physionet', n_subjects=109):
    """Download PhysioNet EEGMMIDB eyes-closed baseline"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://physionet.org/files/eegmmidb/1.0.0/"
    downloaded = 0
    
    for subj in tqdm(range(1, n_subjects + 1), desc="Downloading PhysioNet"):
        subj_str = f"S{subj:03d}"
        filename = f"{subj_str}R02.edf"
        url = base_url + f"{subj_str}/{filename}"
        local_path = save_dir / filename
        
        if local_path.exists():
            downloaded += 1
            continue
        
        try:
            urllib.request.urlretrieve(url, local_path)
            downloaded += 1
        except Exception as e:
            print(f"  Failed {subj_str}: {e}")
    
    print(f"PhysioNet: {downloaded}/{n_subjects} downloaded")
    return save_dir

def get_channel_indices(ch_names, target_channels):
    """Find matching channel indices"""
    indices = []
    matched = []
    for i, ch in enumerate(ch_names):
        ch_clean = ch.upper().replace('.', '').replace('-', '')
        for target in target_channels:
            if target.upper() in ch_clean or ch_clean in target.upper():
                indices.append(i)
                matched.append(ch)
                break
    return indices, matched

def compute_centroids_raw(freqs, psd, theta_band, alpha_band):
    """Compute spectral centroids WITHOUT FOOOF"""
    theta_mask = (freqs >= theta_band[0]) & (freqs <= theta_band[1])
    alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
    
    theta_centroid = np.average(freqs[theta_mask], weights=psd[theta_mask]) if psd[theta_mask].sum() > 0 else np.nan
    alpha_centroid = np.average(freqs[alpha_mask], weights=psd[alpha_mask]) if psd[alpha_mask].sum() > 0 else np.nan
    
    return theta_centroid, alpha_centroid

def compute_centroids_fooof(freqs, psd, theta_band, alpha_band, fooof_settings):
    """Compute spectral centroids WITH FOOOF"""
    try:
        fm = SpectralModel(**fooof_settings)
        fm.fit(freqs, psd, [1, 45])
        corrected_psd = np.maximum(fm._spectrum_flat, 0)
        
        theta_mask = (freqs >= theta_band[0]) & (freqs <= theta_band[1])
        alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
        
        psd_theta = corrected_psd[theta_mask]
        psd_alpha = corrected_psd[alpha_mask]
        
        theta_centroid = np.average(freqs[theta_mask], weights=psd_theta) if psd_theta.sum() > 0 else np.nan
        alpha_centroid = np.average(freqs[alpha_mask], weights=psd_alpha) if psd_alpha.sum() > 0 else np.nan
        
        return theta_centroid, alpha_centroid
    except:
        return np.nan, np.nan

def compute_pci_convergence(theta, alpha, phi=1.618034, eps=0.1):
    """Compute PCI and convergence"""
    if np.isnan(theta) or np.isnan(alpha):
        return np.nan, np.nan, np.nan, np.nan
    
    ratio = alpha / theta
    pci = np.log((abs(ratio - 2.0) + eps) / (abs(ratio - phi) + eps))
    delta_f = abs(alpha - theta)
    conv_orig = 1 / delta_f if delta_f > 0 else np.nan
    conv_bound = 1 / (delta_f + 0.5)
    
    return ratio, pci, conv_orig, conv_bound

def process_subject(raw, config, use_fooof=True):
    """Process single subject"""
    data = raw.get_data()
    fs = raw.info['sfreq']
    ch_names = raw.ch_names
    
    results = {'fs': fs, 'n_channels': len(ch_names)}
    
    for region, target_chs in [('posterior', config['posterior_channels']), 
                                ('frontal', config['frontal_channels'])]:
        
        ch_idx, matched = get_channel_indices(ch_names, target_chs)
        results[f'{region}_channels'] = ','.join(matched)
        
        if len(ch_idx) < 2:
            for suffix in ['_raw', '_fooof'] if use_fooof else ['_raw']:
                results[f'theta_{region}{suffix}'] = np.nan
                results[f'alpha_{region}{suffix}'] = np.nan
            continue
        
        region_data = data[ch_idx].mean(axis=0)
        nperseg = min(int(fs * 2), len(region_data))
        freqs, psd = welch(region_data, fs, nperseg=nperseg)
        
        freq_mask = (freqs >= 1) & (freqs <= 45)
        freqs, psd = freqs[freq_mask], psd[freq_mask]
        
        theta_raw, alpha_raw = compute_centroids_raw(freqs, psd, config['theta_band'], config['alpha_band'])
        results[f'theta_{region}_raw'] = theta_raw
        results[f'alpha_{region}_raw'] = alpha_raw
        
        if use_fooof and HAS_FOOOF:
            theta_fooof, alpha_fooof = compute_centroids_fooof(freqs, psd, config['theta_band'], config['alpha_band'], config['fooof_settings'])
            results[f'theta_{region}_fooof'] = theta_fooof
            results[f'alpha_{region}_fooof'] = alpha_fooof
    
    for method in (['raw', 'fooof'] if use_fooof and HAS_FOOOF else ['raw']):
        for region in ['posterior', 'frontal']:
            theta = results.get(f'theta_{region}_{method}')
            alpha = results.get(f'alpha_{region}_{method}')
            if theta is not None and alpha is not None:
                ratio, pci, conv_orig, conv_bound = compute_pci_convergence(theta, alpha, config['phi'], config['epsilon'])
                results[f'ratio_{region}_{method}'] = ratio
                results[f'pci_{region}_{method}'] = pci
                results[f'conv_orig_{region}_{method}'] = conv_orig
                results[f'conv_bound_{region}_{method}'] = conv_bound
    
    return results

def process_physionet(data_dir='data/physionet', use_fooof=True, checkpoint_every=20):
    """Process PhysioNet with checkpointing"""
    data_dir = Path(data_dir)
    edf_files = sorted(data_dir.glob('*R02.edf'))
    
    print(f"\nProcessing PhysioNet: {len(edf_files)} files")
    
    results = []
    errors = []
    checkpoint_file = Path('results/physionet_checkpoint.csv')
    
    for i, fpath in enumerate(tqdm(edf_files, desc="PhysioNet")):
        try:
            raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
            raw.filter(1, 45, verbose=False)
            
            subj_results = process_subject(raw, CONFIG, use_fooof)
            subj_results['subject'] = fpath.stem
            subj_results['dataset'] = 'PhysioNet'
            subj_results['file'] = str(fpath)
            
            results.append(subj_results)
            
            if (i + 1) % checkpoint_every == 0:
                pd.DataFrame(results).to_csv(checkpoint_file, index=False)
                print(f"  Checkpoint saved: {len(results)} subjects")
            
        except Exception as e:
            errors.append({'file': str(fpath), 'error': str(e)})
    
    df = pd.DataFrame(results)
    print(f"Processed: {len(df)}, Errors: {len(errors)}")
    
    return df, errors

def run_analysis(df, method='raw'):
    """Run statistical analysis"""
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {method.upper()}")
    print(f"{'='*60}")
    
    pci_col = f'pci_posterior_{method}'
    conv_col = f'conv_bound_posterior_{method}'
    
    valid = df[[pci_col, conv_col]].dropna()
    print(f"Valid N: {len(valid)}")
    
    if len(valid) < 10:
        print("Not enough data!")
        return {}
    
    r, p = stats.pearsonr(valid[pci_col], valid[conv_col])
    rho, _ = stats.spearmanr(valid[pci_col], valid[conv_col])
    
    n = len(valid)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    ci_low, ci_high = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    
    phi_org = (valid[pci_col] > 0).sum()
    phi_pct = 100 * phi_org / len(valid)
    
    print(f"\nPearson r = {r:.3f}, p = {p:.2e}")
    print(f"95% CI [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"Spearman ρ = {rho:.3f}")
    print(f"Phi-organized: {phi_org}/{len(valid)} ({phi_pct:.1f}%)")
    
    return {'n': n, 'r': r, 'p': p, 'ci_low': ci_low, 'ci_high': ci_high, 'rho': rho, 'phi_pct': phi_pct}

def main():
    print("="*60)
    print("POD 1: PhysioNet Processing (N=109)")
    print("="*60)
    
    Path('data/physionet').mkdir(parents=True, exist_ok=True)
    Path('results').mkdir(parents=True, exist_ok=True)
    
    print("\n[1] Downloading PhysioNet...")
    download_physionet('data/physionet', n_subjects=109)
    
    print("\n[2] Processing...")
    df, errors = process_physionet('data/physionet', use_fooof=HAS_FOOOF, checkpoint_every=20)
    
    print("\n[3] Saving results...")
    df.to_csv('results/physionet_results.csv', index=False)
    
    print("\n[4] Analysis...")
    results_raw = run_analysis(df, 'raw')
    if HAS_FOOOF:
        results_fooof = run_analysis(df, 'fooof')
    
    print("\n" + "="*60)
    print("COMPLETE! Results saved to results/physionet_results.csv")
    print("="*60)

if __name__ == "__main__":
    main()
