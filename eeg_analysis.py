import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import requests
import zipfile
import io
import os

# Constants
PHI = (1 + np.sqrt(5)) / 2  # 1.618
print(f"Golden Ratio φ = {PHI:.4f}")

# Download EEG Alpha Waves dataset from Zenodo (small, ~50MB)
print("\n📥 Downloading EEG Alpha Waves dataset from Zenodo...")
url = "https://zenodo.org/records/2348892/files/subject_01.mat?download=1"

# Try multiple subjects
subjects_data = []
for subj in range(1, 6):  # Get 5 subjects
    url = f"https://zenodo.org/records/2348892/files/subject_{subj:02d}.mat?download=1"
    print(f"  Downloading subject {subj}...")
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            # Save temporarily
            fname = f"subject_{subj:02d}.mat"
            with open(fname, 'wb') as f:
                f.write(r.content)
            subjects_data.append(fname)
            print(f"    ✓ Downloaded {fname}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

print(f"\n✓ Downloaded {len(subjects_data)} subjects")

# Load and analyze
from scipy.io import loadmat

BANDS = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
}

def compute_peak_centroid(psd, freqs, band):
    """Spectral centroid - robust peak estimation."""
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask) or np.sum(psd[mask]) == 0:
        return np.nan
    return np.sum(freqs[mask] * psd[mask]) / np.sum(psd[mask])

def phi_coupling_index(ratio):
    """PCI: +1 = φ-coupled, -1 = 2:1 coupled."""
    if np.isnan(ratio) or ratio <= 0:
        return np.nan
    d_phi = abs(ratio - PHI)
    d_harm = abs(ratio - 2.0)
    if d_phi + d_harm == 0:
        return 0
    return (d_harm - d_phi) / (d_harm + d_phi)

all_results = []
sfreq = 512  # Alpha Waves dataset sampling rate

print("\n🔬 Analyzing EEG data for φ-switching...")
print("="*60)

for fname in subjects_data:
    try:
        mat = loadmat(fname)
        # Get EEG data (structure varies, try common keys)
        for key in mat.keys():
            if not key.startswith('_'):
                data = mat[key]
                if isinstance(data, np.ndarray) and data.ndim >= 2:
                    # Use first channel or average
                    if data.shape[0] < data.shape[1]:
                        eeg = np.mean(data, axis=0)
                    else:
                        eeg = np.mean(data, axis=1)
                    break
        
        # Compute PSD
        freqs, psd = signal.welch(eeg, sfreq, nperseg=1024)
        
        # Get peak frequencies
        f_theta = compute_peak_centroid(psd, freqs, BANDS['theta'])
        f_alpha = compute_peak_centroid(psd, freqs, BANDS['alpha'])
        f_beta = compute_peak_centroid(psd, freqs, BANDS['beta'])
        
        # Compute ratios and PCI
        ratio_ta = f_alpha / f_theta if f_theta > 0 else np.nan
        ratio_ab = f_beta / f_alpha if f_alpha > 0 else np.nan
        pci_ta = phi_coupling_index(ratio_ta)
        pci_ab = phi_coupling_index(ratio_ab)
        
        result = {
            'subject': fname,
            'f_theta': f_theta,
            'f_alpha': f_alpha,
            'f_beta': f_beta,
            'ratio_ta': ratio_ta,
            'ratio_ab': ratio_ab,
            'pci_ta': pci_ta,
            'pci_ab': pci_ab
        }
        all_results.append(result)
        
        print(f"\n{fname}:")
        print(f"  θ={f_theta:.2f}Hz, α={f_alpha:.2f}Hz, β={f_beta:.2f}Hz")
        print(f"  θ-α ratio: {ratio_ta:.3f} (φ={PHI:.3f})")
        print(f"  PCI: {pci_ta:+.3f}")
        
    except Exception as e:
        print(f"  Error with {fname}: {e}")

# Summary
if all_results:
    print("\n" + "="*60)
    print("📊 SUMMARY - REAL EEG DATA RESULTS")
    print("="*60)
    
    pcis = [r['pci_ta'] for r in all_results if not np.isnan(r['pci_ta'])]
    ratios = [r['ratio_ta'] for r in all_results if not np.isnan(r['ratio_ta'])]
    
    mean_pci = np.mean(pcis)
    mean_ratio = np.mean(ratios)
    
    print(f"\nSubjects analyzed: {len(all_results)}")
    print(f"Mean θ-α ratio: {mean_ratio:.3f}")
    print(f"Golden ratio φ: {PHI:.3f}")
    print(f"Distance from φ: {abs(mean_ratio - PHI):.3f}")
    print(f"\nMean PCI: {mean_pci:+.3f} ± {np.std(pcis):.3f}")
    
    print("\n" + "="*60)
    print("🎯 INTERPRETATION")
    print("="*60)
    
    if mean_pci > 0.1:
        print(f"✅ Mean PCI ({mean_pci:+.3f}) is POSITIVE")
        print("   → EEG shows φ-organization (toward decoupling)")
        print("   → Supports 'standby/receptive' hypothesis")
    elif mean_pci < -0.1:
        print(f"❌ Mean PCI ({mean_pci:+.3f}) is NEGATIVE")
        print("   → EEG shows 2:1 harmonic organization")
        print("   → Indicates 'engaged/processing' state")
    else:
        print(f"⚪ Mean PCI ({mean_pci:+.3f}) is NEAR ZERO")
        print("   → EEG shows intermediate organization")
        
    # Compare to φ
    if abs(mean_ratio - PHI) < abs(mean_ratio - 2.0):
        print(f"\n✓ Ratio ({mean_ratio:.3f}) is CLOSER to φ ({PHI:.3f}) than to 2.0")
    else:
        print(f"\n✗ Ratio ({mean_ratio:.3f}) is CLOSER to 2.0 than to φ ({PHI:.3f})")
