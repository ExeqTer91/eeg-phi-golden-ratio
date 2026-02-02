import numpy as np
from scipy import signal
from scipy.stats import ttest_ind, pearsonr, spearmanr
from scipy.io import loadmat
import matplotlib.pyplot as plt
import requests
import os
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf

mne.set_log_level('ERROR')

PHI = (1 + np.sqrt(5)) / 2
print(f"φ = {PHI:.4f}")
print("="*70)
print("COMPREHENSIVE φ-SWITCHING ANALYSIS")
print("="*70)

BANDS = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}

def get_peak_freq(psd, freqs, band):
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

def analyze_signal(eeg, sfreq):
    freqs, psd = signal.welch(eeg, sfreq, nperseg=min(1024, len(eeg)//2))
    f_theta = get_peak_freq(psd, freqs, BANDS['theta'])
    f_alpha = get_peak_freq(psd, freqs, BANDS['alpha'])
    ratio = f_alpha / f_theta if f_theta > 0 else np.nan
    return {'ratio': ratio, 'pci': pci(ratio), 'f_theta': f_theta, 'f_alpha': f_alpha}

ALL_RESULTS = {}

# ============================================
# DATASET 1: Zenodo Alpha Waves (N=20, resting)
# ============================================
print("\n" + "="*70)
print("DATASET 1: Zenodo Alpha Waves - Resting State (eyes open/closed)")
print("="*70)

results_1 = []
for subj in range(1, 21):
    url = f"https://zenodo.org/records/2348892/files/subject_{subj:02d}.mat?download=1"
    fname = f"alpha_s{subj:02d}.mat"
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
                    res = analyze_signal(eeg, 512)
                    res['subject'] = subj
                    res['condition'] = 'resting'
                    results_1.append(res)
                    break
    except:
        continue

if results_1:
    pcis = [r['pci'] for r in results_1 if not np.isnan(r['pci'])]
    print(f"N = {len(pcis)}")
    print(f"Mean PCI = {np.mean(pcis):+.3f} ± {np.std(pcis):.3f}")
    ALL_RESULTS['Zenodo_Resting'] = results_1

# ============================================
# DATASET 2: EEGBCI - REST vs MOTOR IMAGERY (N=20)
# ============================================
print("\n" + "="*70)
print("DATASET 2: EEGBCI - REST vs MOTOR IMAGERY")
print("="*70)

results_rest = []
results_task = []

for subj in range(1, 21):
    try:
        # REST
        raw = read_raw_edf(eegbci.load_data(subj, [1], update_path=True, verbose=False)[0], preload=True, verbose=False)
        raw.filter(1, 45, verbose=False)
        eeg = np.mean(raw.get_data(), axis=0)
        res = analyze_signal(eeg, raw.info['sfreq'])
        res['subject'] = subj
        res['condition'] = 'rest'
        results_rest.append(res)
        
        # MOTOR IMAGERY
        raw = read_raw_edf(eegbci.load_data(subj, [4], update_path=True, verbose=False)[0], preload=True, verbose=False)
        raw.filter(1, 45, verbose=False)
        eeg = np.mean(raw.get_data(), axis=0)
        res = analyze_signal(eeg, raw.info['sfreq'])
        res['subject'] = subj
        res['condition'] = 'motor_imagery'
        results_task.append(res)
    except:
        continue

pcis_rest = [r['pci'] for r in results_rest if not np.isnan(r['pci'])]
pcis_task = [r['pci'] for r in results_task if not np.isnan(r['pci'])]

print(f"REST: N={len(pcis_rest)}, Mean PCI = {np.mean(pcis_rest):+.3f} ± {np.std(pcis_rest):.3f}")
print(f"TASK: N={len(pcis_task)}, Mean PCI = {np.mean(pcis_task):+.3f} ± {np.std(pcis_task):.3f}")

if len(pcis_rest) >= 2 and len(pcis_task) >= 2:
    t, p = ttest_ind(pcis_rest, pcis_task)
    print(f"T-test: t={t:.3f}, p={p:.4f}")

ALL_RESULTS['EEGBCI_REST'] = results_rest
ALL_RESULTS['EEGBCI_TASK'] = results_task

# ============================================
# DATASET 3: EEGBCI Extended (subjects 21-50)
# ============================================
print("\n" + "="*70)
print("DATASET 3: EEGBCI Extended - More subjects (21-50)")
print("="*70)

results_ext = []
for subj in range(21, 51):
    try:
        raw = read_raw_edf(eegbci.load_data(subj, [1], update_path=True, verbose=False)[0], preload=True, verbose=False)
        raw.filter(1, 45, verbose=False)
        eeg = np.mean(raw.get_data(), axis=0)
        res = analyze_signal(eeg, raw.info['sfreq'])
        res['subject'] = subj
        res['condition'] = 'rest'
        results_ext.append(res)
    except:
        continue

pcis_ext = [r['pci'] for r in results_ext if not np.isnan(r['pci'])]
print(f"N = {len(pcis_ext)}, Mean PCI = {np.mean(pcis_ext):+.3f} ± {np.std(pcis_ext):.3f}")
ALL_RESULTS['EEGBCI_Extended'] = results_ext

# ============================================
# DATASET 4: MNE Sample (auditory/visual task)
# ============================================
print("\n" + "="*70)
print("DATASET 4: MNE Sample Dataset - Auditory/Visual Task")
print("="*70)

try:
    sample_path = mne.datasets.sample.data_path()
    raw = mne.io.read_raw_fif(sample_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif', 
                              preload=True, verbose=False)
    raw.pick_types(eeg=True, exclude='bads')
    raw.filter(1, 45, verbose=False)
    eeg = np.mean(raw.get_data(), axis=0)
    res = analyze_signal(eeg, raw.info['sfreq'])
    res['subject'] = 1
    res['condition'] = 'audvis_task'
    print(f"N = 1, PCI = {res['pci']:+.3f}, ratio = {res['ratio']:.3f}")
    ALL_RESULTS['MNE_AudVis'] = [res]
except Exception as e:
    print(f"Error: {e}")

# ============================================
# GRAND SUMMARY
# ============================================
print("\n" + "="*70)
print("📊 GRAND SUMMARY - ALL DATASETS")
print("="*70)

all_pcis = []
all_conditions = []

summary_table = []
for name, results in ALL_RESULTS.items():
    pcis = [r['pci'] for r in results if not np.isnan(r['pci'])]
    ratios = [r['ratio'] for r in results if not np.isnan(r['ratio'])]
    if pcis:
        summary_table.append({
            'Dataset': name,
            'N': len(pcis),
            'Mean_Ratio': np.mean(ratios),
            'Mean_PCI': np.mean(pcis),
            'Std_PCI': np.std(pcis),
            'Dist_from_phi': abs(np.mean(ratios) - PHI)
        })
        all_pcis.extend(pcis)

print(f"\n{'Dataset':<20} {'N':>5} {'Mean Ratio':>12} {'Mean PCI':>12} {'Dist φ':>10}")
print("-"*65)
for row in summary_table:
    print(f"{row['Dataset']:<20} {row['N']:>5} {row['Mean_Ratio']:>12.3f} {row['Mean_PCI']:>+12.3f} {row['Dist_from_phi']:>10.3f}")

print("\n" + "="*70)
print("🎯 OVERALL STATISTICS")
print("="*70)
print(f"Total N = {len(all_pcis)}")
print(f"Grand Mean PCI = {np.mean(all_pcis):+.3f} ± {np.std(all_pcis):.3f}")
print(f"PCI Range: {min(all_pcis):+.3f} to {max(all_pcis):+.3f}")

# Count by direction
phi_count = sum(1 for p in all_pcis if p > 0)
harm_count = sum(1 for p in all_pcis if p < 0)
print(f"\nφ-organized (PCI > 0): {phi_count} ({100*phi_count/len(all_pcis):.1f}%)")
print(f"2:1 organized (PCI < 0): {harm_count} ({100*harm_count/len(all_pcis):.1f}%)")

# ============================================
# VISUALIZATION
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: PCI by dataset
ax1 = axes[0, 0]
datasets = [row['Dataset'] for row in summary_table]
means = [row['Mean_PCI'] for row in summary_table]
stds = [row['Std_PCI'] for row in summary_table]
colors = ['green' if m > 0 else 'red' for m in means]
bars = ax1.bar(range(len(datasets)), means, yerr=stds, capsize=5, color=colors, alpha=0.7)
ax1.axhline(0, color='black', linestyle='-', linewidth=1)
ax1.set_xticks(range(len(datasets)))
ax1.set_xticklabels([d.replace('_', '\n') for d in datasets], fontsize=9)
ax1.set_ylabel('Mean PCI')
ax1.set_title('A. Phi Coupling Index by Dataset')
ax1.set_ylim(-0.5, 0.5)

# Panel B: Distribution of all PCI values
ax2 = axes[0, 1]
ax2.hist(all_pcis, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(0, color='black', linestyle='--', linewidth=2, label='Neutral')
ax2.axvline(np.mean(all_pcis), color='red', linestyle='-', linewidth=2, label=f'Mean={np.mean(all_pcis):.3f}')
ax2.set_xlabel('PCI')
ax2.set_ylabel('Count')
ax2.set_title('B. Distribution of PCI Values (All Subjects)')
ax2.legend()

# Panel C: Individual variability
ax3 = axes[1, 0]
all_individual = []
for name, results in ALL_RESULTS.items():
    for r in results:
        if not np.isnan(r['pci']):
            all_individual.append({'dataset': name, 'pci': r['pci'], 'ratio': r['ratio']})

ratios_all = [r['ratio'] for r in all_individual]
pcis_all = [r['pci'] for r in all_individual]
ax3.scatter(ratios_all, pcis_all, alpha=0.6, c='steelblue')
ax3.axhline(0, color='black', linestyle='--')
ax3.axvline(PHI, color='gold', linestyle='-', linewidth=2, label=f'φ = {PHI:.3f}')
ax3.axvline(2.0, color='red', linestyle='-', linewidth=2, label='2:1 = 2.0')
ax3.set_xlabel('θ-α Ratio')
ax3.set_ylabel('PCI')
ax3.set_title('C. Individual Subject Ratios vs PCI')
ax3.legend()
ax3.set_xlim(1.4, 2.2)

# Panel D: Summary text
ax4 = axes[1, 1]
ax4.axis('off')
summary_text = f"""
COMPREHENSIVE φ-SWITCHING ANALYSIS RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Subjects Analyzed: {len(all_pcis)}

Key Findings:
• Grand Mean PCI = {np.mean(all_pcis):+.3f} ± {np.std(all_pcis):.3f}
• {phi_count}/{len(all_pcis)} subjects ({100*phi_count/len(all_pcis):.0f}%) show φ-organization
• {harm_count}/{len(all_pcis)} subjects ({100*harm_count/len(all_pcis):.0f}%) show 2:1 organization

Interpretation:
{'✅ OVERALL TENDENCY TOWARD φ' if np.mean(all_pcis) > 0 else '❌ OVERALL TENDENCY TOWARD 2:1'}

The high individual variability (std ≈ {np.std(all_pcis):.2f}) suggests
PCI may be a stable individual trait rather than 
a state-dependent variable.

Golden Ratio φ = {PHI:.4f}
"""
ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('comprehensive_phi_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ Figure saved: comprehensive_phi_analysis.png")
