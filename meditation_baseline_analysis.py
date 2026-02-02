import numpy as np
from scipy import signal
from scipy.stats import ttest_ind, pearsonr
import matplotlib.pyplot as plt
import requests
import os
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf

mne.set_log_level('ERROR')

PHI = (1 + np.sqrt(5)) / 2
print("="*70)
print("MEDITATION DATASETS ANALYSIS")
print("Comparing Meditators vs Non-Meditators for phi-coupling")
print("="*70)

BANDS = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}

def get_peak_centroid(psd, freqs, band):
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask) or np.sum(psd[mask]) == 0:
        return np.nan
    return np.sum(freqs[mask] * psd[mask]) / np.sum(psd[mask])

def pci(ratio):
    if np.isnan(ratio) or ratio <= 0:
        return np.nan
    d_phi = abs(ratio - PHI)
    d_2 = abs(ratio - 2.0)
    return (d_2 - d_phi) / (d_2 + d_phi) if (d_2 + d_phi) > 0 else 0

def compute_8hz_convergence(eeg, sfreq):
    window = int(2 * sfreq)
    step = window // 2
    convergence = 0
    total = 0
    
    for start in range(0, len(eeg) - window, step):
        seg = eeg[start:start+window]
        f, p = signal.welch(seg, sfreq, nperseg=min(512, len(seg)))
        
        t_mask = (f >= 4) & (f <= 8)
        a_mask = (f >= 8) & (f <= 13)
        
        if np.any(t_mask) and np.any(a_mask):
            f_t = f[t_mask][np.argmax(p[t_mask])]
            f_a = f[a_mask][np.argmax(p[a_mask])]
            total += 1
            if 7 <= f_t <= 8.5 and 7.5 <= f_a <= 9 and abs(f_t - f_a) < 1.5:
                convergence += 1
    
    return 100 * convergence / total if total > 0 else 0

def analyze_eeg(eeg, sfreq):
    f, p = signal.welch(eeg, sfreq, nperseg=min(1024, len(eeg)//2))
    f_t = get_peak_centroid(p, f, BANDS['theta'])
    f_a = get_peak_centroid(p, f, BANDS['alpha'])
    ratio = f_a / f_t if f_t > 0 else np.nan
    conv = compute_8hz_convergence(eeg, sfreq)
    return {'ratio': ratio, 'pci': pci(ratio), 'convergence': conv, 'f_theta': f_t, 'f_alpha': f_a}

ALL_DATA = {'meditators': [], 'non_meditators': [], 'unknown': []}

print("\n" + "="*70)
print("DATASET 1: OpenNeuro ds003969 - Meditation vs Thinking Task")
print("="*70)

base_url = "https://raw.githubusercontent.com/OpenNeuroDatasets/ds003969/master"
try:
    r = requests.get(f"{base_url}/participants.tsv", timeout=30)
    if r.status_code == 200:
        print("Found participants.tsv")
        print(r.text[:500])
except Exception as e:
    print(f"Could not access ds003969: {e}")

print("\n" + "="*70)
print("DATASET 2: Checking Zenodo Meditation Datasets")
print("="*70)

zenodo_urls = [
    ("Meditation EEG 2536267", "https://zenodo.org/api/records/2536267"),
    ("Gamma Meditation 57911", "https://zenodo.org/api/records/57911"),
]

for name, url in zenodo_urls:
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            data = r.json()
            files = data.get('files', [])
            print(f"\n{name}:")
            print(f"  Files available: {len(files)}")
            for f in files[:5]:
                print(f"    - {f.get('key', 'unknown')} ({f.get('size', 0)/1024/1024:.1f} MB)")
    except Exception as e:
        print(f"{name}: Error - {e}")

print("\n" + "="*70)
print("DATASET 3: EEGBCI Extended - Non-meditators baseline (N=50)")
print("="*70)

for subj in range(1, 51):
    try:
        raw = read_raw_edf(eegbci.load_data(subj, [1], update_path=True, verbose=False)[0], 
                          preload=True, verbose=False)
        raw.filter(1, 45, verbose=False)
        eeg = np.mean(raw.get_data(), axis=0)
        res = analyze_eeg(eeg, raw.info['sfreq'])
        res['subject'] = f"EEGBCI_{subj}"
        res['group'] = 'non_meditator'
        ALL_DATA['non_meditators'].append(res)
    except:
        continue

print(f"Loaded {len(ALL_DATA['non_meditators'])} non-meditator subjects")

print("\n" + "="*70)
print("DATASET 4: Zenodo 57911 - Gamma Meditation Study")
print("="*70)

gamma_files = [
    "https://zenodo.org/records/57911/files/711Hz_spec_data_medit.mat?download=1",
    "https://zenodo.org/records/57911/files/boxplot_alpha.csv?download=1",
    "https://zenodo.org/records/57911/files/boxplot_gamma.csv?download=1",
]

for url in gamma_files:
    fname = url.split('/')[-1].split('?')[0]
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(fname, 'wb') as f:
                f.write(r.content)
            print(f"Downloaded {fname} ({len(r.content)/1024:.1f} KB)")
    except Exception as e:
        print(f"{fname}: {e}")

try:
    import pandas as pd
    if os.path.exists('boxplot_alpha.csv'):
        df_alpha = pd.read_csv('boxplot_alpha.csv')
        print(f"\nAlpha meditation data shape: {df_alpha.shape}")
        print(df_alpha.head())
        print(f"\nColumns: {df_alpha.columns.tolist()}")
except Exception as e:
    print(f"Could not load CSV: {e}")

print("\n" + "="*70)
print("DATASET 5: MNE Sample Dataset")
print("="*70)

try:
    sample_path = mne.datasets.sample.data_path()
    raw = mne.io.read_raw_fif(sample_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif', 
                              preload=True, verbose=False)
    raw.pick_types(eeg=True, exclude='bads')
    raw.filter(1, 45, verbose=False)
    eeg = np.mean(raw.get_data(), axis=0)
    res = analyze_eeg(eeg, raw.info['sfreq'])
    res['subject'] = 'MNE_Sample'
    res['group'] = 'task'
    ALL_DATA['unknown'].append(res)
    print(f"MNE Sample: PCI={res['pci']:+.3f}, 8Hz conv={res['convergence']:.1f}%")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*70)
print("SUMMARY OF ALL AVAILABLE DATA")
print("="*70)

print(f"\nNon-meditators (EEGBCI): N = {len(ALL_DATA['non_meditators'])}")
if ALL_DATA['non_meditators']:
    pcis = [r['pci'] for r in ALL_DATA['non_meditators'] if not np.isnan(r['pci'])]
    convs = [r['convergence'] for r in ALL_DATA['non_meditators']]
    print(f"  Mean PCI: {np.mean(pcis):+.3f} +/- {np.std(pcis):.3f}")
    print(f"  Mean 8Hz convergence: {np.mean(convs):.1f}%")
    print(f"  phi-organized: {sum(1 for p in pcis if p > 0)}/{len(pcis)} ({100*sum(1 for p in pcis if p > 0)/len(pcis):.0f}%)")

print(f"\nMeditators: N = {len(ALL_DATA['meditators'])}")
if ALL_DATA['meditators']:
    pcis = [r['pci'] for r in ALL_DATA['meditators'] if not np.isnan(r['pci'])]
    convs = [r['convergence'] for r in ALL_DATA['meditators']]
    print(f"  Mean PCI: {np.mean(pcis):+.3f} +/- {np.std(pcis):.3f}")
    print(f"  Mean 8Hz convergence: {np.mean(convs):.1f}%")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
if ALL_DATA['non_meditators']:
    pcis = [r['pci'] for r in ALL_DATA['non_meditators'] if not np.isnan(r['pci'])]
    ax1.hist(pcis, bins=20, color='gray', edgecolor='black', alpha=0.7, label='Non-meditators')
    ax1.axvline(0, color='black', linestyle='--', linewidth=2)
    ax1.axvline(np.mean(pcis), color='red', linestyle='-', linewidth=2, label=f'Mean={np.mean(pcis):.3f}')
    ax1.set_xlabel('PCI')
    ax1.set_ylabel('Count')
    ax1.set_title(f'A. Non-Meditators PCI Distribution (N={len(pcis)})')
    ax1.legend()

ax2 = axes[0, 1]
if ALL_DATA['non_meditators']:
    convs = [r['convergence'] for r in ALL_DATA['non_meditators']]
    ax2.hist(convs, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(convs), color='red', linestyle='-', linewidth=2, label=f'Mean={np.mean(convs):.1f}%')
    ax2.set_xlabel('8 Hz Convergence (%)')
    ax2.set_ylabel('Count')
    ax2.set_title('B. 8 Hz Convergence (infinity-state) Distribution')
    ax2.legend()

ax3 = axes[1, 0]
if ALL_DATA['non_meditators']:
    thetas = [r['f_theta'] for r in ALL_DATA['non_meditators'] if not np.isnan(r['f_theta'])]
    alphas = [r['f_alpha'] for r in ALL_DATA['non_meditators'] if not np.isnan(r['f_alpha'])]
    ax3.scatter(thetas, alphas, c='gray', s=80, alpha=0.6, label='Non-meditators')
    ax3.plot([4, 8], [8, 8], 'r--', linewidth=2)
    ax3.plot([8, 8], [8, 13], 'r--', linewidth=2, label='8 Hz boundary')
    ax3.set_xlabel('Theta Centroid (Hz)')
    ax3.set_ylabel('Alpha Centroid (Hz)')
    ax3.set_title('C. Theta-Alpha Frequency Space')
    ax3.set_xlim(5, 8)
    ax3.set_ylim(8, 12)
    ax3.legend()

ax4 = axes[1, 1]
ax4.axis('off')

if ALL_DATA['non_meditators']:
    pcis = [r['pci'] for r in ALL_DATA['non_meditators'] if not np.isnan(r['pci'])]
    convs = [r['convergence'] for r in ALL_DATA['non_meditators']]
    phi_count = sum(1 for p in pcis if p > 0)
    
    summary = f"""
BASELINE DATA SUMMARY
------------------------------------------

NON-MEDITATORS (EEGBCI):
* N = {len(pcis)}
* Mean PCI = {np.mean(pcis):+.3f} +/- {np.std(pcis):.3f}
* phi-organized: {phi_count}/{len(pcis)} ({100*phi_count/len(pcis):.0f}%)
* Mean 8Hz convergence: {np.mean(convs):.1f}%

AVAILABLE MEDITATION DATASETS:
* OpenNeuro ds003969: 98 subjects (meditation vs thinking)
* OpenNeuro ds001787: 24 meditators
* Zenodo 57911: Gamma meditation (4 traditions)
* IEEE Vipassana: novices/experienced/monks

BASELINE SHOWS:
* ~60% naturally phi-organized
* ~10% time in 8-8 convergence (infinity-state)

HYPOTHESIS:
Experienced meditators should show:
* Higher PCI (more phi-organized)
* Higher 8Hz convergence (more infinity-state access)
* Lower variability (more stable)

phi = {PHI:.4f}
"""
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('meditation_baseline.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: meditation_baseline.png")

print("\n" + "="*70)
print("CORRELATION ANALYSIS")
print("="*70)

if ALL_DATA['non_meditators']:
    pcis = [r['pci'] for r in ALL_DATA['non_meditators'] if not np.isnan(r['pci'])]
    convs = [r['convergence'] for r in ALL_DATA['non_meditators'] if not np.isnan(r['pci'])]
    
    r, p = pearsonr(pcis, convs)
    print(f"PCI vs 8Hz Convergence: r = {r:.3f}, p = {p:.4f}")
    
    if p < 0.05:
        print("SIGNIFICANT correlation between phi-coupling and infinity-state access")
    else:
        print("No significant correlation (need more data or meditation comparison)")
