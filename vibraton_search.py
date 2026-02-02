"""
VIBRATON SEARCH
===============
Looking for discrete quantized peaks in θ/α ratio distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from scipy import signal
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf

mne.set_log_level('ERROR')

PHI = (1 + np.sqrt(5)) / 2

print("="*70)
print("VIBRATON SEARCH")
print("Looking for discrete quantized states in θ/α ratio distribution")
print("="*70)

BANDS = {'theta': (4, 8), 'alpha': (8, 13)}

def get_peak_centroid(psd, freqs, band):
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask) or np.sum(psd[mask]) == 0:
        return np.nan
    return np.sum(freqs[mask] * psd[mask]) / np.sum(psd[mask])

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

print("\nLoading data from 50 subjects...")

ratios = []
convergences = []

for subj in range(1, 51):
    try:
        raw = read_raw_edf(eegbci.load_data(subj, [1], update_path=True, verbose=False)[0], 
                          preload=True, verbose=False)
        raw.filter(1, 45, verbose=False)
        sfreq = raw.info['sfreq']
        eeg = np.mean(raw.get_data(), axis=0)
        
        f, p = signal.welch(eeg, sfreq, nperseg=min(1024, len(eeg)//2))
        f_t = get_peak_centroid(p, f, BANDS['theta'])
        f_a = get_peak_centroid(p, f, BANDS['alpha'])
        
        if f_t > 0 and not np.isnan(f_t) and not np.isnan(f_a):
            ratio = f_a / f_t
            conv = compute_8hz_convergence(eeg, sfreq)
            ratios.append(ratio)
            convergences.append(conv)
    except:
        continue

ratios = np.array(ratios)
convergences = np.array(convergences)

print(f"Loaded {len(ratios)} subjects")
print(f"Mean ratio: {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")
print(f"Range: [{np.min(ratios):.4f}, {np.max(ratios):.4f}]")

print("\n" + "="*70)
print("VIBRATON ANALYSIS - Multiple Bandwidths")
print("="*70)

bandwidths = [0.05, 0.08, 0.1, 0.12, 0.15]
x = np.linspace(1.5, 2.1, 500)

all_peaks = []

for bw in bandwidths:
    kde = gaussian_kde(ratios, bw_method=bw)
    y = kde(x)
    
    peaks, properties = find_peaks(y, prominence=0.05)
    
    print(f"\nBandwidth = {bw}:")
    print(f"  Found {len(peaks)} peaks:")
    for p in peaks:
        all_peaks.append(x[p])
        print(f"    Ratio = {x[p]:.4f}, prominence = {properties['prominences'][np.where(peaks == p)[0][0]]:.3f}")

print("\n" + "="*70)
print("VIBRATON CANDIDATES SUMMARY")
print("="*70)

if all_peaks:
    all_peaks = np.array(all_peaks)
    
    from scipy.cluster.hierarchy import fcluster, linkage
    
    if len(all_peaks) > 1:
        linkage_matrix = linkage(all_peaks.reshape(-1, 1), method='average')
        clusters = fcluster(linkage_matrix, t=0.05, criterion='distance')
        
        unique_clusters = np.unique(clusters)
        vibraton_candidates = []
        
        print("\nClustered VIBRATON candidates:")
        for c in unique_clusters:
            cluster_peaks = all_peaks[clusters == c]
            mean_peak = np.mean(cluster_peaks)
            vibraton_candidates.append(mean_peak)
            print(f"  VIBRATON {c}: ratio = {mean_peak:.4f} ({len(cluster_peaks)} detections)")
        
        vibraton_candidates = sorted(vibraton_candidates)
        
        print(f"\nFinal VIBRATON candidates: {len(vibraton_candidates)}")
        for i, v in enumerate(vibraton_candidates):
            dist_phi = abs(v - PHI)
            dist_2 = abs(v - 2.0)
            print(f"  V{i+1} = {v:.4f} (dist from φ: {dist_phi:.4f}, dist from 2:1: {dist_2:.4f})")
        
        if len(vibraton_candidates) > 1:
            spacings = np.diff(vibraton_candidates)
            print(f"\nSpacings between VIBRATONs: {spacings}")
            print(f"Mean spacing: {np.mean(spacings):.4f}")
            
            print("\nSpacing ratios:")
            for i, s in enumerate(spacings):
                print(f"  Spacing {i+1}: {s:.4f}")
                print(f"    vs φ-1: {s / (PHI - 1):.3f}")
                print(f"    vs 1/φ: {s * PHI:.3f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
kde = gaussian_kde(ratios, bw_method=0.1)
y = kde(x)
peaks, _ = find_peaks(y, prominence=0.05)

ax1.plot(x, y, 'b-', linewidth=2)
ax1.plot(x[peaks], y[peaks], 'ro', markersize=12, label='Peaks')
ax1.axvline(x=PHI, color='gold', linestyle='--', linewidth=2, label=f'φ = {PHI:.3f}')
ax1.axvline(x=2.0, color='red', linestyle='--', linewidth=2, label='2:1 = 2.0')
ax1.set_xlabel('θ/α Ratio', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('A. VIBRATON Search - KDE Peaks (bw=0.1)', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.hist(ratios, bins=20, density=True, alpha=0.7, color='steelblue', edgecolor='black')
for v in vibraton_candidates:
    ax2.axvline(x=v, color='purple', linestyle='-', linewidth=2, alpha=0.7)
ax2.axvline(x=PHI, color='gold', linestyle='--', linewidth=2, label='φ')
ax2.set_xlabel('θ/α Ratio', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title('B. Histogram with VIBRATON markers', fontsize=14)
ax2.legend(fontsize=10)

ax3 = axes[1, 0]
colors = ['green' if c > 15 else 'orange' if c > 5 else 'gray' for c in convergences]
ax3.scatter(ratios, convergences, c=colors, s=80, alpha=0.7, edgecolor='black')
for v in vibraton_candidates:
    ax3.axvline(x=v, color='purple', linestyle='-', linewidth=2, alpha=0.5, label=f'V={v:.3f}')
ax3.axvline(x=PHI, color='gold', linestyle='--', linewidth=2, label='φ')
ax3.set_xlabel('θ/α Ratio', fontsize=12)
ax3.set_ylabel('8 Hz Convergence (%)', fontsize=12)
ax3.set_title('C. Individual subjects with VIBRATON lines', fontsize=14)
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
ax4.axis('off')

summary = f"""
VIBRATON SEARCH SUMMARY
================================================

N = {len(ratios)} subjects
Ratio range: [{np.min(ratios):.4f}, {np.max(ratios):.4f}]
Mean ratio: {np.mean(ratios):.4f} ± {np.std(ratios):.4f}

VIBRATON CANDIDATES FOUND: {len(vibraton_candidates)}
"""

for i, v in enumerate(vibraton_candidates):
    summary += f"\n  V{i+1} = {v:.4f}"

if len(vibraton_candidates) > 1:
    spacings = np.diff(vibraton_candidates)
    summary += f"\n\nSpacings: {[f'{s:.4f}' for s in spacings]}"
    summary += f"\nMean spacing: {np.mean(spacings):.4f}"

summary += f"""

INTERPRETATION:
Peaks in the ratio distribution suggest
possible QUANTIZED STATES (vibrations) in
the θ/α frequency relationship.

Key reference points:
  φ = {PHI:.4f}
  2:1 = 2.0000
"""

ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('vibraton_search.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: vibraton_search.png")

print("\n" + "="*70)
print("VIBRATON vs HARMONIC RATIOS")
print("="*70)

harmonic_ratios = {
    '3:2': 1.5,
    '8:5': 1.6,
    'φ': PHI,
    '5:3': 1.667,
    '7:4': 1.75,
    '9:5': 1.8,
    '2:1': 2.0
}

print("\nVibraton candidates vs known harmonic ratios:")
for v in vibraton_candidates:
    nearest_name = min(harmonic_ratios.keys(), key=lambda k: abs(harmonic_ratios[k] - v))
    nearest_val = harmonic_ratios[nearest_name]
    print(f"  V = {v:.4f} closest to {nearest_name} ({nearest_val:.4f}), distance = {abs(v - nearest_val):.4f}")
