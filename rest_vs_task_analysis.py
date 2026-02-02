import numpy as np
from scipy import signal
from scipy.io import loadmat
from scipy.stats import ttest_ind, mannwhitneyu
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

# ============================================
# MNE's EEGBCI dataset (motor imagery)
# ============================================
print("="*60)
print("MNE EEGBCI Dataset (Rest vs Task)")
print("="*60)

import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

mne.set_log_level('WARNING')

results = {'rest': [], 'task': []}

for subject in range(1, 11):
    print(f"\nSubject {subject}:")
    
    try:
        rest_files = eegbci.load_data(subject, [1], path=None, update_path=True, verbose=False)
        raw_rest = concatenate_raws([
            read_raw_edf(f, preload=True, verbose=False) 
            for f in rest_files
        ])
        raw_rest.filter(1, 45, verbose=False)
        
        task_files = eegbci.load_data(subject, [4], path=None, update_path=True, verbose=False)
        raw_task = concatenate_raws([
            read_raw_edf(f, preload=True, verbose=False) 
            for f in task_files
        ])
        raw_task.filter(1, 45, verbose=False)
        
        sfreq = raw_rest.info['sfreq']
        
        eeg_rest = np.mean(raw_rest.get_data(), axis=0)
        freqs, psd = signal.welch(eeg_rest, sfreq, nperseg=1024)
        f_theta = compute_peak_centroid(psd, freqs, BANDS['theta'])
        f_alpha = compute_peak_centroid(psd, freqs, BANDS['alpha'])
        ratio = f_alpha / f_theta if f_theta > 0 else np.nan
        pci = phi_coupling_index(ratio)
        results['rest'].append({'subject': subject, 'ratio': ratio, 'pci': pci})
        print(f"  REST:  ratio={ratio:.3f}, PCI={pci:+.3f}")
        
        eeg_task = np.mean(raw_task.get_data(), axis=0)
        freqs, psd = signal.welch(eeg_task, sfreq, nperseg=1024)
        f_theta = compute_peak_centroid(psd, freqs, BANDS['theta'])
        f_alpha = compute_peak_centroid(psd, freqs, BANDS['alpha'])
        ratio = f_alpha / f_theta if f_theta > 0 else np.nan
        pci = phi_coupling_index(ratio)
        results['task'].append({'subject': subject, 'ratio': ratio, 'pci': pci})
        print(f"  TASK:  ratio={ratio:.3f}, PCI={pci:+.3f}")
        
    except Exception as e:
        print(f"  Error: {e}")

# ============================================
# STATISTICAL COMPARISON
# ============================================
print("\n" + "="*60)
print("📊 STATISTICAL COMPARISON: REST vs TASK")
print("="*60)

rest_pcis = [r['pci'] for r in results['rest'] if not np.isnan(r['pci'])]
task_pcis = [r['pci'] for r in results['task'] if not np.isnan(r['pci'])]
rest_ratios = [r['ratio'] for r in results['rest'] if not np.isnan(r['ratio'])]
task_ratios = [r['ratio'] for r in results['task'] if not np.isnan(r['ratio'])]

print(f"\nREST (eyes open):")
print(f"  N = {len(rest_pcis)}")
print(f"  Mean θ-α ratio: {np.mean(rest_ratios):.3f} ± {np.std(rest_ratios):.3f}")
print(f"  Mean PCI: {np.mean(rest_pcis):+.3f} ± {np.std(rest_pcis):.3f}")

print(f"\nTASK (motor imagery):")
print(f"  N = {len(task_pcis)}")
print(f"  Mean θ-α ratio: {np.mean(task_ratios):.3f} ± {np.std(task_ratios):.3f}")
print(f"  Mean PCI: {np.mean(task_pcis):+.3f} ± {np.std(task_pcis):.3f}")

if len(rest_pcis) >= 2 and len(task_pcis) >= 2:
    t_stat, p_value = ttest_ind(rest_pcis, task_pcis)
    print(f"\nT-test (REST vs TASK PCI):")
    print(f"  t = {t_stat:.3f}, p = {p_value:.4f}")
    
    pooled_std = np.sqrt((np.std(rest_pcis)**2 + np.std(task_pcis)**2) / 2)
    cohens_d = (np.mean(rest_pcis) - np.mean(task_pcis)) / pooled_std if pooled_std > 0 else 0
    print(f"  Cohen's d = {cohens_d:.3f}")
    
    if p_value < 0.05:
        print(f"\n✅ SIGNIFICANT DIFFERENCE (p < 0.05)")
    else:
        print(f"\n⚪ No significant difference (p = {p_value:.3f})")

print("\n" + "="*60)
print("🎯 INTERPRETATION")
print("="*60)

rest_dist_phi = abs(np.mean(rest_ratios) - PHI)
task_dist_phi = abs(np.mean(task_ratios) - PHI)

print(f"\nDistance from φ ({PHI:.3f}):")
print(f"  REST: {rest_dist_phi:.3f}")
print(f"  TASK: {task_dist_phi:.3f}")

if rest_dist_phi < task_dist_phi:
    print(f"\n→ REST is CLOSER to φ (more decoupled)")
else:
    print(f"\n→ TASK is CLOSER to φ (more decoupled)")

print("\n" + "="*60)
print("📈 INDIVIDUAL SUBJECT DATA")
print("="*60)
print("\n{:<10} {:>12} {:>12} {:>12} {:>12}".format("Subject", "REST ratio", "REST PCI", "TASK ratio", "TASK PCI"))
print("-" * 60)
for i, (r, t) in enumerate(zip(results['rest'], results['task']), 1):
    print(f"{i:<10} {r['ratio']:>12.3f} {r['pci']:>+12.3f} {t['ratio']:>12.3f} {t['pci']:>+12.3f}")
