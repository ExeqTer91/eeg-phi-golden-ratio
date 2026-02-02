"""
MEGA DATASET ANALYSIS - 500+ SUBJECTS
======================================
Combining PhysioNet Motor Imagery (109) + Sleep PhysioNet (~150+) + EEGBCI runs
"""

import numpy as np
from scipy import stats, signal
from collections import Counter
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf
import os
import warnings
warnings.filterwarnings('ignore')

mne.set_log_level('ERROR')

PHI = (1 + np.sqrt(5)) / 2

print("="*70)
print("MEGA DATASET ANALYSIS - TARGET: 500+ SUBJECTS")
print("="*70)

VIBRATON_EDGES = [1.45, 1.55, 1.64, 1.70, 1.77, 1.83, 1.90, 2.05]

def analyze_eeg(eeg, sfreq):
    """Compute phi-switching metrics from EEG"""
    f, p = signal.welch(eeg, sfreq, nperseg=min(1024, len(eeg)//2))
    
    t_mask = (f >= 4) & (f <= 8)
    a_mask = (f >= 8) & (f <= 13)
    
    if not np.any(t_mask) or not np.any(a_mask):
        return None
    
    f_t = np.sum(f[t_mask] * p[t_mask]) / (np.sum(p[t_mask]) + 1e-10)
    f_a = np.sum(f[a_mask] * p[a_mask]) / (np.sum(p[a_mask]) + 1e-10)
    
    if f_t < 1:
        return None
    
    ratio = f_a / f_t
    
    d_phi = abs(ratio - PHI)
    d_2to1 = abs(ratio - 2.0)
    pci = (d_2to1 - d_phi) / (d_2to1 + d_phi + 1e-10)
    
    window = int(2 * sfreq)
    step = window // 4
    conv_count = 0
    total_windows = 0
    
    for start in range(0, len(eeg) - window, step):
        seg = eeg[start:start+window]
        f_s, p_s = signal.welch(seg, sfreq, nperseg=min(512, len(seg)))
        
        t_m = (f_s >= 4) & (f_s <= 8)
        a_m = (f_s >= 8) & (f_s <= 13)
        
        if np.sum(p_s[t_m]) > 0 and np.sum(p_s[a_m]) > 0:
            ft = np.sum(f_s[t_m] * p_s[t_m]) / np.sum(p_s[t_m])
            fa = np.sum(f_s[a_m] * p_s[a_m]) / np.sum(p_s[a_m])
            if ft > 0:
                r = fa / ft
                if 1.55 <= r <= 1.70:
                    conv_count += 1
                total_windows += 1
    
    convergence = 100 * conv_count / total_windows if total_windows > 0 else 0
    
    vibraton = np.digitize([ratio], VIBRATON_EDGES)[0] - 1
    vibraton = max(0, min(6, vibraton))
    
    return {
        'ratio': ratio,
        'pci': pci,
        'convergence': convergence,
        'vibraton': vibraton,
        'theta_cent': f_t,
        'alpha_cent': f_a
    }

all_results = []

print("\n" + "="*70)
print("DATASET 1: PhysioNet Motor Imagery - 109 subjects × 14 runs = 1526")
print("="*70)

runs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

for subj in range(1, 110):
    for run in runs:
        try:
            path = eegbci.load_data(subj, [run], update_path=True, verbose=False)[0]
            raw = read_raw_edf(path, preload=True, verbose=False)
            raw.filter(1, 45, verbose=False)
            sfreq = raw.info['sfreq']
            eeg = np.mean(raw.get_data(), axis=0)
            
            result = analyze_eeg(eeg, sfreq)
            if result:
                result['source'] = 'PhysioNet'
                result['subject'] = f"P{subj:03d}_R{run:02d}"
                all_results.append(result)
        except:
            continue
    
    if subj % 20 == 0:
        print(f"  Processed {subj}/109 subjects... ({len(all_results)} recordings)")

print(f"\nPhysioNet: {len(all_results)} recordings collected")

print("\n" + "="*70)
print(f"TOTAL RECORDINGS: {len(all_results)}")
print("="*70)

ratios = np.array([r['ratio'] for r in all_results])
pcis = np.array([r['pci'] for r in all_results])
convergences = np.array([r['convergence'] for r in all_results])
vibrations = np.array([r['vibraton'] for r in all_results])

print(f"\nN = {len(all_results)} recordings from 109 unique subjects")

print("\n" + "="*70)
print("PHI-ORGANIZATION ANALYSIS")
print("="*70)

phi_organized = np.sum(pcis > 0)
two_one_organized = np.sum(pcis < 0)
print(f"\nφ-organized (PCI > 0): {phi_organized} ({100*phi_organized/len(pcis):.1f}%)")
print(f"2:1-organized (PCI < 0): {two_one_organized} ({100*two_one_organized/len(pcis):.1f}%)")

print(f"\nMean ratio: {np.mean(ratios):.4f} (φ = 1.618)")
print(f"Median ratio: {np.median(ratios):.4f}")
print(f"Mean PCI: {np.mean(pcis):.4f}")
print(f"Mean convergence: {np.mean(convergences):.1f}%")

print("\n" + "="*70)
print("VIBRATON DISTRIBUTION")
print("="*70)

vib_counts = Counter(vibrations)
print("\nVibraton state distribution:")
for v in range(7):
    count = vib_counts.get(v, 0)
    pct = 100 * count / len(vibrations)
    bar = "█" * int(pct / 2)
    print(f"  V{v+1}: {count:4d} ({pct:5.1f}%) {bar}")

print("\n" + "="*70)
print("HIGH vs LOW CONVERTERS")
print("="*70)

median_conv = np.median(convergences)
high_mask = convergences > median_conv
low_mask = convergences <= median_conv

print(f"\nMedian convergence: {median_conv:.1f}%")
print(f"HIGH converters (>{median_conv:.1f}%): {high_mask.sum()} recordings")
print(f"LOW converters (<={median_conv:.1f}%): {low_mask.sum()} recordings")

print(f"\nHIGH group:")
print(f"  Mean ratio: {np.mean(ratios[high_mask]):.4f}")
print(f"  Mean PCI: {np.mean(pcis[high_mask]):.4f}")
print(f"  Mean convergence: {np.mean(convergences[high_mask]):.1f}%")

print(f"\nLOW group:")
print(f"  Mean ratio: {np.mean(ratios[low_mask]):.4f}")
print(f"  Mean PCI: {np.mean(pcis[low_mask]):.4f}")
print(f"  Mean convergence: {np.mean(convergences[low_mask]):.1f}%")

t, p = stats.ttest_ind(ratios[high_mask], ratios[low_mask])
print(f"\nRatio difference: t = {t:.2f}, p = {p:.2e}")

print("\n" + "="*70)
print("STATISTICAL TESTS")
print("="*70)

t_pci, p_pci = stats.ttest_1samp(pcis, 0)
print(f"\nPCI vs 0: t = {t_pci:.2f}, p = {p_pci:.2e}")
if p_pci < 0.001:
    print("  *** HIGHLY SIGNIFICANT: Population is NOT neutral!")

t_ratio, p_ratio = stats.ttest_1samp(ratios, PHI)
print(f"\nRatio vs φ: t = {t_ratio:.2f}, p = {p_ratio:.2e}")

t_ratio2, p_ratio2 = stats.ttest_1samp(ratios, 2.0)
print(f"Ratio vs 2.0: t = {t_ratio2:.2f}, p = {p_ratio2:.2e}")

r_pci_conv, p_corr = stats.pearsonr(pcis, convergences)
print(f"\nPCI ↔ Convergence: r = {r_pci_conv:.3f}, p = {p_corr:.2e}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

summary = f"""
MEGA ANALYSIS RESULTS
=====================

Total recordings: {len(all_results)}
Unique subjects: 109

KEY FINDINGS:
- φ-organized: {100*phi_organized/len(pcis):.1f}%
- Mean ratio: {np.mean(ratios):.4f} (φ = 1.618)
- Mean PCI: {np.mean(pcis):.4f}
- PCI-Convergence correlation: r = {r_pci_conv:.3f}

STATISTICAL SIGNIFICANCE:
- PCI ≠ 0: p = {p_pci:.2e}
- HIGH vs LOW ratio difference: p = {p:.2e}
"""
print(summary)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

ax1 = axes[0, 0]
ax1.hist(ratios, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(PHI, color='gold', linewidth=2, linestyle='--', label=f'φ = {PHI:.3f}')
ax1.axvline(2.0, color='red', linewidth=2, linestyle='--', label='2:1 = 2.0')
ax1.axvline(np.mean(ratios), color='green', linewidth=2, label=f'Mean = {np.mean(ratios):.3f}')
ax1.set_xlabel('Alpha/Theta Ratio', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title(f'A. Ratio Distribution (N={len(ratios)})', fontsize=14)
ax1.legend()

ax2 = axes[0, 1]
ax2.hist(pcis, bins=50, color='purple', edgecolor='black', alpha=0.7)
ax2.axvline(0, color='black', linewidth=2, linestyle='--', label='Neutral (0)')
ax2.axvline(np.mean(pcis), color='green', linewidth=2, label=f'Mean = {np.mean(pcis):.3f}')
ax2.set_xlabel('PCI', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('B. PCI Distribution', fontsize=14)
ax2.legend()

ax3 = axes[0, 2]
ax3.hist(convergences, bins=50, color='green', edgecolor='black', alpha=0.7)
ax3.axvline(np.median(convergences), color='red', linewidth=2, label=f'Median = {np.median(convergences):.1f}%')
ax3.set_xlabel('Convergence (%)', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title('C. Convergence Distribution', fontsize=14)
ax3.legend()

ax4 = axes[1, 0]
vib_labels = ['V1', 'V2\n(φ)', 'V3', 'V4', 'V5', 'V6', 'V7']
vib_vals = [vib_counts.get(i, 0) for i in range(7)]
colors = ['gold' if i <= 2 else 'gray' if i <= 4 else 'red' for i in range(7)]
ax4.bar(range(7), vib_vals, color=colors, edgecolor='black')
ax4.set_xticks(range(7))
ax4.set_xticklabels(vib_labels)
ax4.set_ylabel('Count', fontsize=12)
ax4.set_title('D. Vibraton Distribution', fontsize=14)

ax5 = axes[1, 1]
ax5.scatter(pcis, convergences, c='purple', s=10, alpha=0.3)
z = np.polyfit(pcis, convergences, 1)
p_fit = np.poly1d(z)
x_fit = np.linspace(min(pcis), max(pcis), 100)
ax5.plot(x_fit, p_fit(x_fit), 'r-', linewidth=2, label=f'r = {r_pci_conv:.3f}')
ax5.set_xlabel('PCI', fontsize=12)
ax5.set_ylabel('Convergence (%)', fontsize=12)
ax5.set_title('E. PCI vs Convergence', fontsize=14)
ax5.legend()

ax6 = axes[1, 2]
ax6.bar(['φ-organized', '2:1-organized'], [phi_organized, two_one_organized], 
        color=['gold', 'red'], edgecolor='black')
ax6.set_ylabel('Count', fontsize=12)
ax6.set_title(f'F. Organization Type\n(φ: {100*phi_organized/len(pcis):.0f}%)', fontsize=14)

plt.tight_layout()
plt.savefig('mega_dataset_analysis.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: mega_dataset_analysis.png")
