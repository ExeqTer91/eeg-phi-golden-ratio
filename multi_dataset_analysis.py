import numpy as np
from scipy import signal
from scipy.io import loadmat
import matplotlib.pyplot as plt
import requests
import os

PHI = (1 + np.sqrt(5)) / 2
print(f"Golden Ratio φ = {PHI:.4f}\n")

BANDS = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}

def compute_peak_centroid(psd, freqs, band):
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask) or np.sum(psd[mask]) == 0:
        return np.nan
    return np.sum(freqs[mask] * psd[mask]) / np.sum(psd[mask])

def phi_coupling_index(ratio):
    if np.isnan(ratio) or ratio <= 0:
        return np.nan
    d_phi = abs(ratio - PHI)
    d_harm = abs(ratio - 2.0)
    if d_phi + d_harm == 0:
        return 0
    return (d_harm - d_phi) / (d_harm + d_phi)

def analyze_eeg(eeg, sfreq, label=""):
    """Analyze single EEG signal."""
    freqs, psd = signal.welch(eeg, sfreq, nperseg=min(1024, len(eeg)//2))
    f_theta = compute_peak_centroid(psd, freqs, BANDS['theta'])
    f_alpha = compute_peak_centroid(psd, freqs, BANDS['alpha'])
    ratio = f_alpha / f_theta if f_theta > 0 else np.nan
    pci = phi_coupling_index(ratio)
    return {'label': label, 'f_theta': f_theta, 'f_alpha': f_alpha, 'ratio': ratio, 'pci': pci}

all_datasets = {}

# ============================================
# DATASET 1: Alpha Waves (eyes open/closed) - 20 subjects
# ============================================
print("="*60)
print("DATASET 1: Alpha Waves (Zenodo) - Resting State")
print("="*60)

results_1 = []
for subj in range(1, 21):
    url = f"https://zenodo.org/records/2348892/files/subject_{subj:02d}.mat?download=1"
    fname = f"alpha_subj_{subj:02d}.mat"
    try:
        if not os.path.exists(fname):
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(fname, 'wb') as f:
                    f.write(r.content)
        
        mat = loadmat(fname)
        for key in mat.keys():
            if not key.startswith('_'):
                data = mat[key]
                if isinstance(data, np.ndarray) and data.ndim >= 2:
                    eeg = np.mean(data[:, 1:17], axis=1) if data.shape[1] > 17 else np.mean(data, axis=1)
                    result = analyze_eeg(eeg, 512, f"S{subj:02d}")
                    results_1.append(result)
                    break
    except Exception as e:
        continue

if results_1:
    pcis = [r['pci'] for r in results_1 if not np.isnan(r['pci'])]
    ratios = [r['ratio'] for r in results_1 if not np.isnan(r['ratio'])]
    print(f"N = {len(results_1)}")
    print(f"Mean θ-α ratio: {np.mean(ratios):.3f} ± {np.std(ratios):.3f}")
    print(f"Mean PCI: {np.mean(pcis):+.3f} ± {np.std(pcis):.3f}")
    all_datasets['Alpha Waves\n(Resting)'] = {'pcis': pcis, 'ratios': ratios}

# ============================================
# DATASET 2: Meditation EEG (Zenodo 57911) - 4 groups
# ============================================
print("\n" + "="*60)
print("DATASET 2: Meditation Gamma Study (3 traditions + control)")
print("="*60)

# This dataset has preprocessed spectral data
url_spec = "https://zenodo.org/records/57911/files/711Hz_spec_data_medit.mat?download=1"
try:
    r = requests.get(url_spec, timeout=30)
    if r.status_code == 200:
        with open('meditation_spec.mat', 'wb') as f:
            f.write(r.content)
        mat = loadmat('meditation_spec.mat')
        print("Available keys:", [k for k in mat.keys() if not k.startswith('_')])
        
        # Try to extract spectrum data
        for key in mat.keys():
            if not key.startswith('_'):
                data = mat[key]
                print(f"  {key}: shape {data.shape if hasattr(data, 'shape') else type(data)}")
except Exception as e:
    print(f"Could not load: {e}")

# ============================================
# DATASET 3: MNE Sample Data (built-in)
# ============================================
print("\n" + "="*60)
print("DATASET 3: MNE Sample Dataset (Auditory/Visual)")
print("="*60)

try:
    import mne
    # Use MNE's built-in sample data
    sample_data_folder = mne.datasets.sample.data_path()
    raw_fname = sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
    raw.pick_types(eeg=True, exclude='bads')
    raw.filter(1, 45, verbose=False)
    
    sfreq = raw.info['sfreq']
    data = raw.get_data()
    eeg = np.mean(data, axis=0)
    
    result = analyze_eeg(eeg, sfreq, "MNE Sample")
    print(f"θ-α ratio: {result['ratio']:.3f}")
    print(f"PCI: {result['pci']:+.3f}")
    all_datasets['MNE Sample\n(Task)'] = {'pcis': [result['pci']], 'ratios': [result['ratio']]}
except Exception as e:
    print(f"MNE sample failed: {e}")

# ============================================
# SUMMARY ACROSS ALL DATASETS
# ============================================
print("\n" + "="*60)
print("📊 CROSS-DATASET SUMMARY")
print("="*60)

for name, data in all_datasets.items():
    mean_pci = np.mean(data['pcis'])
    mean_ratio = np.mean(data['ratios'])
    n = len(data['pcis'])
    print(f"\n{name.replace(chr(10), ' ')}:")
    print(f"  N = {n}")
    print(f"  Mean ratio = {mean_ratio:.3f} (φ={PHI:.3f}, 2:1=2.000)")
    print(f"  Mean PCI = {mean_pci:+.3f}")
    
    if mean_ratio < PHI:
        print(f"  → Ratio BELOW φ")
    elif mean_ratio < 2.0:
        print(f"  → Ratio BETWEEN φ and 2:1")
    else:
        print(f"  → Ratio ABOVE 2:1")

# Combine all
all_pcis = []
all_ratios = []
for data in all_datasets.values():
    all_pcis.extend(data['pcis'])
    all_ratios.extend(data['ratios'])

print("\n" + "="*60)
print("🎯 GRAND TOTAL (All Datasets Combined)")
print("="*60)
print(f"Total N = {len(all_pcis)}")
print(f"Grand Mean θ-α ratio: {np.mean(all_ratios):.3f} ± {np.std(all_ratios):.3f}")
print(f"Grand Mean PCI: {np.mean(all_pcis):+.3f} ± {np.std(all_pcis):.3f}")
print(f"\nφ = {PHI:.3f}")
print(f"Distance from φ: {abs(np.mean(all_ratios) - PHI):.3f}")
print(f"Distance from 2.0: {abs(np.mean(all_ratios) - 2.0):.3f}")

if abs(np.mean(all_ratios) - PHI) < abs(np.mean(all_ratios) - 2.0):
    print("\n✅ OVERALL: EEG ratios are CLOSER to φ than to 2:1")
else:
    print("\n❌ OVERALL: EEG ratios are CLOSER to 2:1 than to φ")
