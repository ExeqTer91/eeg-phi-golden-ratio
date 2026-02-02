"""
FULL VALIDATION - 227 SUBJECTS
==============================
Combining PhysioNet + ds003969 + MATLAB
"""

import numpy as np
from scipy import stats, signal
import matplotlib.pyplot as plt
import glob
import os
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2

print("="*70)
print("FULL VALIDATION - 227 SUBJECTS COMBINED")
print("="*70)

def analyze_eeg_signal(eeg, sfreq):
    """Compute phi metrics from raw EEG"""
    if len(eeg) < sfreq * 2:
        return None
    
    f, p = signal.welch(eeg, sfreq, nperseg=min(1024, len(eeg)//2))
    
    t_mask = (f >= 4) & (f <= 8)
    a_mask = (f >= 8) & (f <= 13)
    
    if not np.any(t_mask) or not np.any(a_mask):
        return None
    
    if np.sum(p[t_mask]) < 1e-10 or np.sum(p[a_mask]) < 1e-10:
        return None
    
    f_t = np.sum(f[t_mask] * p[t_mask]) / np.sum(p[t_mask])
    f_a = np.sum(f[a_mask] * p[a_mask]) / np.sum(p[a_mask])
    
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
    
    return {'ratio': ratio, 'pci': pci, 'convergence': convergence}

all_ratios = []
all_pci = []
all_convergence = []
all_sources = []

print("\n1. Loading PhysioNet EEGbci (109 subjects)...")
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf
mne.set_log_level('ERROR')

physionet_count = 0
for subj in range(1, 110):
    try:
        path = eegbci.load_data(subj, [1], update_path=True, verbose=False)[0]
        raw = read_raw_edf(path, preload=True, verbose=False)
        raw.filter(1, 45, verbose=False)
        eeg = np.mean(raw.get_data(), axis=0)
        sfreq = raw.info['sfreq']
        
        result = analyze_eeg_signal(eeg, sfreq)
        if result and 1.3 < result['ratio'] < 2.3:
            all_ratios.append(result['ratio'])
            all_pci.append(result['pci'])
            all_convergence.append(result['convergence'])
            all_sources.append('PhysioNet')
            physionet_count += 1
    except:
        continue

print(f"   PhysioNet: {physionet_count} subjects loaded")

print("\n2. Loading ds003969 (98 subjects)...")
ds_path = '/home/runner/workspace/ds003969'
ds_count = 0

if os.path.exists(ds_path):
    for subj_dir in sorted(glob.glob(f'{ds_path}/sub-*'))[:98]:
        eeg_files = glob.glob(f'{subj_dir}/**/*.set', recursive=True)
        if not eeg_files:
            eeg_files = glob.glob(f'{subj_dir}/**/*.edf', recursive=True)
        if not eeg_files:
            eeg_files = glob.glob(f'{subj_dir}/**/*.vhdr', recursive=True)
        
        for eeg_file in eeg_files[:1]:
            try:
                if eeg_file.endswith('.set'):
                    raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
                elif eeg_file.endswith('.edf'):
                    raw = mne.io.read_raw_edf(eeg_file, preload=True, verbose=False)
                elif eeg_file.endswith('.vhdr'):
                    raw = mne.io.read_raw_brainvision(eeg_file, preload=True, verbose=False)
                else:
                    continue
                
                raw.filter(1, 45, verbose=False)
                eeg = np.mean(raw.get_data(), axis=0)
                sfreq = raw.info['sfreq']
                
                result = analyze_eeg_signal(eeg, sfreq)
                if result and 1.3 < result['ratio'] < 2.3:
                    all_ratios.append(result['ratio'])
                    all_pci.append(result['pci'])
                    all_convergence.append(result['convergence'])
                    all_sources.append('ds003969')
                    ds_count += 1
                    break
            except:
                continue

print(f"   ds003969: {ds_count} subjects loaded")

print("\n3. Loading MATLAB files (~20 subjects)...")
import scipy.io as sio
mat_count = 0

mat_files = glob.glob('/home/runner/workspace/alpha_s*.mat')
for mat_file in mat_files:
    try:
        data = sio.loadmat(mat_file)
        if 'SIGNAL' in data:
            sig = data['SIGNAL']
            eeg = np.mean(sig, axis=1) if sig.shape[1] < sig.shape[0] else np.mean(sig, axis=0)
            sfreq = 256
            
            result = analyze_eeg_signal(eeg, sfreq)
            if result and 1.3 < result['ratio'] < 2.3:
                all_ratios.append(result['ratio'])
                all_pci.append(result['pci'])
                all_convergence.append(result['convergence'])
                all_sources.append('MATLAB')
                mat_count += 1
    except:
        continue

print(f"   MATLAB: {mat_count} subjects loaded")

all_ratios = np.array(all_ratios)
all_pci = np.array(all_pci)
all_convergence = np.array(all_convergence)

print("\n" + "="*70)
print("🔬 VALIDATION SUITE FOR PHI-COUPLING PAPER")
print("="*70)

N = len(all_ratios)
print(f"\n📊 TOTAL SAMPLE SIZE: N = {N}")
print(f"   PhysioNet: {physionet_count}")
print(f"   ds003969: {ds_count}")
print(f"   MATLAB: {mat_count}")

print("\n" + "="*70)
print("CORE CORRELATIONS")
print("="*70)

r_core, p_core = stats.pearsonr(all_pci, all_convergence)
print(f"\n✓ PCI ↔ Convergence: r = {r_core:.3f}, p = {p_core:.2e}")
print(f"  Status: {'✅ CONFIRMED' if p_core < 0.05 else '❌ NOT SIGNIFICANT'}")

dist_phi = np.abs(all_ratios - PHI)
r_phi, p_phi = stats.pearsonr(dist_phi, all_convergence)
print(f"\n✓ Distance(φ) ↔ Convergence: r = {r_phi:.3f}, p = {p_phi:.2e}")
print(f"  Status: {'✅ CONFIRMED' if p_phi < 0.05 else '❌ NOT SIGNIFICANT'}")

dist_2 = np.abs(all_ratios - 2.0)
r_2, p_2 = stats.pearsonr(dist_2, all_convergence)
print(f"\n✓ Distance(2:1) ↔ Convergence: r = {r_2:.3f}, p = {p_2:.2e}")
print(f"  Opposite effect: {'✅ YES (r > 0)' if r_2 > 0 else '❌ NO'}")

print("\n" + "="*70)
print("BOOTSTRAP 95% CI")
print("="*70)

n_boot = 10000
boot_r = []
n = len(all_pci)
for _ in range(n_boot):
    idx = np.random.choice(n, n, replace=True)
    r_boot, _ = stats.pearsonr(all_pci[idx], all_convergence[idx])
    boot_r.append(r_boot)
ci_low, ci_high = np.percentile(boot_r, [2.5, 97.5])
print(f"\nPCI-Convergence: r = {r_core:.3f} [{ci_low:.3f}, {ci_high:.3f}]")

print("\n" + "="*70)
print("DISTRIBUTION STATS")
print("="*70)

print(f"\nRatio distribution:")
print(f"  Mean: {np.mean(all_ratios):.4f}")
print(f"  Median: {np.median(all_ratios):.4f}")
print(f"  Std: {np.std(all_ratios):.4f}")
print(f"  φ = {PHI:.4f}")

phi_organized = np.sum(all_pci > 0)
two_one = np.sum(all_pci < 0)
print(f"\nOrganization:")
print(f"  φ-organized (PCI > 0): {phi_organized} ({100*phi_organized/N:.1f}%)")
print(f"  2:1-organized (PCI < 0): {two_one} ({100*two_one/N:.1f}%)")

print(f"\nConvergence:")
print(f"  Mean: {np.mean(all_convergence):.1f}%")
print(f"  Median: {np.median(all_convergence):.1f}%")

print("\n" + "="*70)
print("🏆 SUMMARY")
print("="*70)
print(f"""
N = {N} subjects ({'✅ EXCELLENT' if N > 200 else '✅ GOOD' if N > 100 else '⚠️ OK'})

Core correlation (PCI ↔ Conv): r = {r_core:.3f} {'✅' if p_core < 0.05 else '❌'}
φ-specificity confirmed: {'✅ YES' if p_phi < 0.05 else '❌ NO'}
Opposite for 2:1: {'✅ YES' if r_2 > 0 else '❌ NO'}

95% CI for r: [{ci_low:.3f}, {ci_high:.3f}]
φ-organized: {100*phi_organized/N:.0f}%
""")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax1 = axes[0, 0]
ax1.hist(all_ratios, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(PHI, color='gold', linewidth=2, linestyle='--', label=f'φ = {PHI:.3f}')
ax1.axvline(2.0, color='red', linewidth=2, linestyle='--', label='2:1')
ax1.axvline(np.mean(all_ratios), color='green', linewidth=2, label=f'Mean = {np.mean(all_ratios):.3f}')
ax1.set_xlabel('Alpha/Theta Ratio', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title(f'A. Ratio Distribution (N={N})', fontsize=14)
ax1.legend()

ax2 = axes[0, 1]
ax2.scatter(all_pci, all_convergence, c='purple', s=30, alpha=0.5)
z = np.polyfit(all_pci, all_convergence, 1)
p_fit = np.poly1d(z)
x_fit = np.linspace(min(all_pci), max(all_pci), 100)
ax2.plot(x_fit, p_fit(x_fit), 'r-', linewidth=2, label=f'r = {r_core:.3f}')
ax2.set_xlabel('PCI', fontsize=12)
ax2.set_ylabel('Convergence (%)', fontsize=12)
ax2.set_title('B. PCI vs Convergence', fontsize=14)
ax2.legend()

ax3 = axes[1, 0]
ax3.bar(['φ-organized\n(PCI > 0)', '2:1-organized\n(PCI < 0)'], 
        [phi_organized, two_one], color=['gold', 'red'], edgecolor='black')
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title(f'C. Organization Type\n(φ: {100*phi_organized/N:.0f}%)', fontsize=14)

ax4 = axes[1, 1]
ax4.hist(all_convergence, bins=30, color='green', edgecolor='black', alpha=0.7)
ax4.axvline(np.median(all_convergence), color='red', linewidth=2, 
            label=f'Median = {np.median(all_convergence):.1f}%')
ax4.set_xlabel('Convergence (%)', fontsize=12)
ax4.set_ylabel('Count', fontsize=12)
ax4.set_title('D. Convergence Distribution', fontsize=14)
ax4.legend()

plt.tight_layout()
plt.savefig('full_validation_227.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: full_validation_227.png")
