"""
SPLIT-HALF VALIDATION SCRIPT
============================
This script validates that PCI-Convergence correlation is NOT a mathematical artifact.

Method:
- For each subject, split EEG epochs into ODD and EVEN
- Calculate PCI from ODD epochs
- Calculate Convergence from EVEN epochs  
- Correlate PCI_odd with Convergence_even

If r > 0.3 and p < 0.001, the relationship is a STABLE TRAIT, not circular math.

Run this in your local environment with the data files.
"""

import numpy as np
from scipy import stats
from scipy.io import loadmat
import os
import glob

# ============================================================
# CONFIGURATION - UPDATE THESE PATHS TO YOUR LOCAL DATA
# ============================================================
DATA_DIR = "."  # Current directory
OUTPUT_FILE = "split_half_results.txt"

# Frequency bands
THETA_BAND = (4, 8)
ALPHA_BAND = (8, 13)
PHI = 1.618033988749895

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def compute_spectral_centroid(psd, freqs, band):
    """Compute power-weighted mean frequency within a band."""
    mask = (freqs >= band[0]) & (freqs <= band[1])
    band_freqs = freqs[mask]
    band_power = psd[mask]
    
    if band_power.sum() == 0:
        return np.nan
    
    centroid = np.sum(band_freqs * band_power) / np.sum(band_power)
    return centroid

def compute_pci(theta_centroid, alpha_centroid):
    """Compute Phi Coupling Index."""
    if np.isnan(theta_centroid) or np.isnan(alpha_centroid):
        return np.nan
    
    ratio = alpha_centroid / theta_centroid
    
    # Avoid division by zero
    dist_phi = np.abs(ratio - PHI) + 1e-10
    dist_2 = np.abs(ratio - 2.0) + 1e-10
    
    pci = np.log(dist_2 / dist_phi)
    return pci

def compute_convergence(theta_centroid, alpha_centroid):
    """Compute theta-alpha convergence (inverse of separation)."""
    if np.isnan(theta_centroid) or np.isnan(alpha_centroid):
        return np.nan
    
    separation = np.abs(alpha_centroid - theta_centroid)
    if separation == 0:
        return np.nan
    
    convergence = 1.0 / separation
    return convergence

def welch_psd(signal, fs, nperseg=None):
    """Compute PSD using Welch method."""
    from scipy.signal import welch
    
    if nperseg is None:
        nperseg = min(4 * fs, len(signal))  # 4 second windows
    
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    return freqs, psd

# ============================================================
# SPLIT-HALF VALIDATION
# ============================================================

def split_half_validation_single_subject(epochs, fs):
    """
    For a single subject with multiple epochs:
    - ODD epochs -> compute PCI
    - EVEN epochs -> compute Convergence
    
    Returns: pci_odd, convergence_even
    """
    n_epochs = len(epochs)
    
    if n_epochs < 4:
        return np.nan, np.nan
    
    odd_epochs = epochs[::2]   # 0, 2, 4, ...
    even_epochs = epochs[1::2]  # 1, 3, 5, ...
    
    # Compute PCI from ODD epochs (average PSD first)
    odd_psds = []
    freqs_odd = None
    for epoch in odd_epochs:
        if len(epoch) > fs:  # At least 1 second
            freqs_odd, psd = welch_psd(epoch, fs)
            odd_psds.append(psd)
    
    if len(odd_psds) == 0 or freqs_odd is None:
        return np.nan, np.nan
    
    avg_psd_odd = np.mean(odd_psds, axis=0)
    theta_odd = compute_spectral_centroid(avg_psd_odd, freqs_odd, THETA_BAND)
    alpha_odd = compute_spectral_centroid(avg_psd_odd, freqs_odd, ALPHA_BAND)
    pci_odd = compute_pci(theta_odd, alpha_odd)
    
    # Compute Convergence from EVEN epochs
    even_psds = []
    freqs_even = None
    for epoch in even_epochs:
        if len(epoch) > fs:
            freqs_even, psd = welch_psd(epoch, fs)
            even_psds.append(psd)
    
    if len(even_psds) == 0 or freqs_even is None:
        return np.nan, np.nan
    
    avg_psd_even = np.mean(even_psds, axis=0)
    theta_even = compute_spectral_centroid(avg_psd_even, freqs_even, THETA_BAND)
    alpha_even = compute_spectral_centroid(avg_psd_even, freqs_even, ALPHA_BAND)
    convergence_even = compute_convergence(theta_even, alpha_even)
    
    return pci_odd, convergence_even

def split_half_from_continuous(signal, fs, epoch_length_sec=4):
    """
    Split continuous signal into epochs, then do split-half.
    """
    epoch_samples = int(epoch_length_sec * fs)
    n_epochs = len(signal) // epoch_samples
    
    if n_epochs < 4:
        return np.nan, np.nan
    
    epochs = []
    for i in range(n_epochs):
        start = i * epoch_samples
        end = start + epoch_samples
        epochs.append(signal[start:end])
    
    return split_half_validation_single_subject(epochs, fs)

# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_split_half_validation():
    """
    Main function to run split-half validation on all subjects.
    
    ADAPT THIS TO YOUR DATA STRUCTURE!
    """
    
    results = {
        'pci_odd': [],
        'convergence_even': [],
        'subject_ids': []
    }
    
    # --------------------------------------------------------
    # OPTION 1: Load from .mat files (alpha_s01.mat, etc.)
    # --------------------------------------------------------
    mat_files = glob.glob(os.path.join(DATA_DIR, "alpha_s*.mat"))
    mat_files += glob.glob(os.path.join(DATA_DIR, "alpha_subj_*.mat"))
    mat_files += glob.glob(os.path.join(DATA_DIR, "subject_*.mat"))
    
    print(f"Found {len(mat_files)} .mat files")
    
    for mat_file in mat_files:
        try:
            data = loadmat(mat_file)
            
            # Try to find EEG data - adapt keys to your structure
            eeg_data = None
            fs = 256  # Default, update if known
            
            for key in ['SIGNAL', 'data', 'EEG', 'eeg', 'signal', 'x', 'y']:
                if key in data and isinstance(data[key], np.ndarray):
                    eeg_data = data[key]
                    break
            
            if eeg_data is None:
                print(f"  Could not find EEG data in {mat_file}")
                continue
            
            # If multi-channel (samples x channels), average across channels
            if eeg_data.ndim > 1:
                eeg_data = np.mean(eeg_data, axis=1)  # Average across channels (axis=1)
            
            eeg_data = eeg_data.flatten()
            
            # Run split-half
            pci_odd, conv_even = split_half_from_continuous(eeg_data, fs)
            
            if not np.isnan(pci_odd) and not np.isnan(conv_even):
                results['pci_odd'].append(pci_odd)
                results['convergence_even'].append(conv_even)
                results['subject_ids'].append(os.path.basename(mat_file))
                print(f"  {os.path.basename(mat_file)}: PCI={pci_odd:.3f}, Conv={conv_even:.3f}")
            
        except Exception as e:
            print(f"  Error loading {mat_file}: {e}")
    
    # --------------------------------------------------------
    # OPTION 2: Load from .npz files (if you have them)
    # --------------------------------------------------------
    npz_files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
    
    for npz_file in npz_files:
        try:
            data = np.load(npz_file, allow_pickle=True)
            
            # Check for pre-computed results
            if 'theta_centroids' in data and 'alpha_centroids' in data:
                # Can't do true split-half without raw epochs
                # But can bootstrap resample
                print(f"  {npz_file}: Pre-computed centroids found, skipping split-half")
                continue
                
        except Exception as e:
            print(f"  Error loading {npz_file}: {e}")
    
    # --------------------------------------------------------
    # COMPUTE CORRELATION
    # --------------------------------------------------------
    
    pci_arr = np.array(results['pci_odd'])
    conv_arr = np.array(results['convergence_even'])
    
    n_valid = len(pci_arr)
    print(f"\n{'='*60}")
    print(f"SPLIT-HALF VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"N subjects with valid split-half: {n_valid}")
    
    if n_valid < 10:
        print("\nWARNING: Too few subjects for reliable analysis!")
        print("You need raw epoch data for proper split-half validation.")
        print("\nALTERNATIVE: Use bootstrap resampling on your existing centroids.")
        return None
    
    # Pearson correlation
    r, p = stats.pearsonr(pci_arr, conv_arr)
    
    # Bootstrap CI
    n_boot = 10000
    boot_rs = []
    for _ in range(n_boot):
        idx = np.random.choice(n_valid, n_valid, replace=True)
        boot_r, _ = stats.pearsonr(pci_arr[idx], conv_arr[idx])
        boot_rs.append(boot_r)
    
    ci_low, ci_high = np.percentile(boot_rs, [2.5, 97.5])
    
    print(f"\nSPLIT-HALF CORRELATION:")
    print(f"  r = {r:.3f}")
    print(f"  p = {p:.2e}")
    print(f"  95% CI = [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"  N = {n_valid}")
    
    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION:")
    print(f"{'='*60}")
    
    if r > 0.3 and p < 0.001:
        print("SUCCESS! The relationship is a STABLE TRAIT.")
        print("   PCI computed on ODD epochs predicts Convergence on EVEN epochs.")
        print("   This rules out mathematical circularity as the sole explanation.")
        print("\n   ADD TO PAPER:")
        print('   "To rule out analytical circularity, we performed split-half')
        print('    cross-validation. PCI computed from odd epochs correlated')
        print(f'    significantly with convergence from even epochs (r = {r:.3f},')
        print(f'    p = {p:.2e}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]), demonstrating')
        print('    that the association reflects stable individual differences')
        print('    rather than mathematical coupling."')
    elif r > 0.2 and p < 0.05:
        print("MODERATE support. Relationship exists but weaker.")
        print("   Consider reframing claims to be more conservative.")
    else:
        print("FAILED. The relationship may indeed be mathematical artifact.")
        print("   Recommend reframing paper as 'descriptive geometric relationship'.")
    
    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        f.write("SPLIT-HALF VALIDATION RESULTS\n")
        f.write(f"N = {n_valid}\n")
        f.write(f"r = {r:.4f}\n")
        f.write(f"p = {p:.2e}\n")
        f.write(f"95% CI = [{ci_low:.4f}, {ci_high:.4f}]\n")
    
    print(f"\nResults saved to {OUTPUT_FILE}")
    
    return {'r': r, 'p': p, 'ci': (ci_low, ci_high), 'n': n_valid}

# ============================================================
# ALTERNATIVE: Bootstrap validation if no raw epochs
# ============================================================

def bootstrap_validation_from_centroids(theta_centroids, alpha_centroids, n_boot=10000):
    """
    If you only have pre-computed centroids (not raw epochs),
    use bootstrap to estimate stability of the correlation.
    
    This doesn't fully address circularity, but shows robustness.
    """
    n = len(theta_centroids)
    
    # Compute PCI and Convergence for all subjects
    pci = []
    conv = []
    
    for t, a in zip(theta_centroids, alpha_centroids):
        pci.append(compute_pci(t, a))
        conv.append(compute_convergence(t, a))
    
    pci = np.array(pci)
    conv = np.array(conv)
    
    # Remove NaN
    valid = ~(np.isnan(pci) | np.isnan(conv))
    pci = pci[valid]
    conv = conv[valid]
    n_valid = len(pci)
    
    # Original correlation
    r_orig, p_orig = stats.pearsonr(pci, conv)
    
    # Bootstrap
    boot_rs = []
    for _ in range(n_boot):
        idx = np.random.choice(n_valid, n_valid, replace=True)
        r_boot, _ = stats.pearsonr(pci[idx], conv[idx])
        boot_rs.append(r_boot)
    
    ci_low, ci_high = np.percentile(boot_rs, [2.5, 97.5])
    
    print(f"\nBOOTSTRAP VALIDATION (N={n_valid}):")
    print(f"  r = {r_orig:.3f}, p = {p_orig:.2e}")
    print(f"  95% CI = [{ci_low:.3f}, {ci_high:.3f}]")
    
    return {'r': r_orig, 'p': p_orig, 'ci': (ci_low, ci_high)}

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("PHI COUPLING INDEX - SPLIT-HALF VALIDATION")
    print("="*60)
    print(f"\nLooking for data in: {DATA_DIR}")
    print("Make sure to update DATA_DIR to your local path!\n")
    
    # Check if directory exists
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Directory {DATA_DIR} not found!")
        print("Please update DATA_DIR at the top of this script.")
    else:
        run_split_half_validation()
