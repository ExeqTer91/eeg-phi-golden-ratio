import numpy as np
from scipy import signal
from scipy.stats import circmean, pearsonr
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf

mne.set_log_level('ERROR')

PHI = (1 + np.sqrt(5)) / 2
print("="*70)
print("CROSS-FREQUENCY COUPLING ANALYSIS")
print("Measuring theta-alpha Phase Synchronization & 8 Hz Boundary")
print("="*70)

def compute_phase_locking(eeg, sfreq, theta_band=(4,8), alpha_band=(8,13)):
    """Compute phase locking value between theta and alpha."""
    b_theta, a_theta = signal.butter(4, [theta_band[0]/(sfreq/2), theta_band[1]/(sfreq/2)], 'band')
    theta = signal.filtfilt(b_theta, a_theta, eeg)
    
    b_alpha, a_alpha = signal.butter(4, [alpha_band[0]/(sfreq/2), alpha_band[1]/(sfreq/2)], 'band')
    alpha = signal.filtfilt(b_alpha, a_alpha, eeg)
    
    theta_phase = np.angle(signal.hilbert(theta))
    alpha_phase = np.angle(signal.hilbert(alpha))
    
    phase_diff_2_1 = alpha_phase - 2 * theta_phase
    plv_2_1 = np.abs(np.mean(np.exp(1j * phase_diff_2_1)))
    
    phase_diff_phi = alpha_phase - PHI * theta_phase
    plv_phi = np.abs(np.mean(np.exp(1j * phase_diff_phi)))
    
    coupling_ratio = plv_phi / plv_2_1 if plv_2_1 > 0 else np.nan
    
    return {
        'plv_2_1': plv_2_1,
        'plv_phi': plv_phi,
        'coupling_ratio': coupling_ratio,
        'theta_phase': theta_phase,
        'alpha_phase': alpha_phase
    }

def find_8hz_convergence(eeg, sfreq):
    """Detect moments when theta and alpha peaks converge near 8 Hz."""
    window_sec = 2
    window_samples = int(window_sec * sfreq)
    step = window_samples // 2
    
    convergence_times = []
    theta_peaks = []
    alpha_peaks = []
    
    for start in range(0, len(eeg) - window_samples, step):
        segment = eeg[start:start + window_samples]
        freqs, psd = signal.welch(segment, sfreq, nperseg=min(512, len(segment)))
        
        theta_mask = (freqs >= 4) & (freqs <= 8)
        if np.any(theta_mask) and np.sum(psd[theta_mask]) > 0:
            theta_freqs = freqs[theta_mask]
            theta_psd = psd[theta_mask]
            f_theta = theta_freqs[np.argmax(theta_psd)]
        else:
            f_theta = np.nan
            
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        if np.any(alpha_mask) and np.sum(psd[alpha_mask]) > 0:
            alpha_freqs = freqs[alpha_mask]
            alpha_psd = psd[alpha_mask]
            f_alpha = alpha_freqs[np.argmax(alpha_psd)]
        else:
            f_alpha = np.nan
        
        theta_peaks.append(f_theta)
        alpha_peaks.append(f_alpha)
        
        if not np.isnan(f_theta) and not np.isnan(f_alpha):
            if 7 <= f_theta <= 8.5 and 7.5 <= f_alpha <= 9:
                distance = abs(f_theta - f_alpha)
                if distance < 1.5:
                    convergence_times.append(start / sfreq)
    
    return {
        'theta_peaks': theta_peaks,
        'alpha_peaks': alpha_peaks,
        'convergence_times': convergence_times,
        'convergence_percent': 100 * len(convergence_times) / max(1, len(theta_peaks))
    }

print("\nAnalyzing cross-frequency dynamics...")

all_results = []

for subj in range(1, 21):
    try:
        raw = read_raw_edf(eegbci.load_data(subj, [1], update_path=True, verbose=False)[0], 
                          preload=True, verbose=False)
        raw.filter(1, 45, verbose=False)
        sfreq = raw.info['sfreq']
        eeg = np.mean(raw.get_data(), axis=0)
        
        plv_results = compute_phase_locking(eeg, sfreq)
        conv_results = find_8hz_convergence(eeg, sfreq)
        
        result = {
            'subject': subj,
            'plv_2_1': plv_results['plv_2_1'],
            'plv_phi': plv_results['plv_phi'],
            'coupling_ratio': plv_results['coupling_ratio'],
            'convergence_percent': conv_results['convergence_percent'],
            'mean_theta_peak': np.nanmean(conv_results['theta_peaks']),
            'mean_alpha_peak': np.nanmean(conv_results['alpha_peaks'])
        }
        all_results.append(result)
        
        print(f"S{subj:02d}: PLV(2:1)={plv_results['plv_2_1']:.3f}, PLV(phi)={plv_results['plv_phi']:.3f}, "
              f"Ratio={plv_results['coupling_ratio']:.2f}, 8Hz-conv={conv_results['convergence_percent']:.1f}%")
        
    except Exception as e:
        print(f"S{subj:02d}: Error - {e}")
        continue

print("\n" + "="*70)
print("CROSS-FREQUENCY COUPLING SUMMARY")
print("="*70)

plv_2_1_all = [r['plv_2_1'] for r in all_results]
plv_phi_all = [r['plv_phi'] for r in all_results]
ratios_all = [r['coupling_ratio'] for r in all_results]
conv_all = [r['convergence_percent'] for r in all_results]

print(f"\nN = {len(all_results)} subjects")
print(f"\nPhase Locking Values:")
print(f"  PLV (2:1 harmonic): {np.mean(plv_2_1_all):.3f} +/- {np.std(plv_2_1_all):.3f}")
print(f"  PLV (phi:1 golden): {np.mean(plv_phi_all):.3f} +/- {np.std(plv_phi_all):.3f}")
print(f"\nCoupling Ratio (phi/2:1): {np.mean(ratios_all):.3f} +/- {np.std(ratios_all):.3f}")
print(f"  >1 = more phi-coupled, <1 = more 2:1-coupled")

print(f"\n8 Hz Convergence (infinity state):")
print(f"  Mean convergence: {np.mean(conv_all):.1f}% of time")
print(f"  Range: {min(conv_all):.1f}% - {max(conv_all):.1f}%")

phi_dominant = sum(1 for r in ratios_all if r > 1)
harmonic_dominant = sum(1 for r in ratios_all if r < 1)
print(f"\nDominant coupling mode:")
print(f"  phi-dominant (ratio > 1): {phi_dominant}/{len(ratios_all)} subjects ({100*phi_dominant/len(ratios_all):.0f}%)")
print(f"  2:1-dominant (ratio < 1): {harmonic_dominant}/{len(ratios_all)} subjects ({100*harmonic_dominant/len(ratios_all):.0f}%)")

r, p = pearsonr(conv_all, ratios_all)
print(f"\nCorrelation: 8 Hz convergence vs Coupling Ratio:")
print(f"  r = {r:.3f}, p = {p:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
x = np.arange(len(all_results))
width = 0.35
ax1.bar(x - width/2, plv_2_1_all, width, label='PLV 2:1 (harmonic)', color='red', alpha=0.7)
ax1.bar(x + width/2, plv_phi_all, width, label='PLV phi:1 (golden)', color='gold', alpha=0.7)
ax1.set_xlabel('Subject')
ax1.set_ylabel('Phase Locking Value')
ax1.set_title('A. Phase Locking: 2:1 Harmonic vs phi:1 Golden Ratio')
ax1.legend()
ax1.set_xticks(x)
ax1.set_xticklabels([f'S{r["subject"]}' for r in all_results], rotation=45)

ax2 = axes[0, 1]
ax2.hist(ratios_all, bins=15, color='purple', edgecolor='black', alpha=0.7)
ax2.axvline(1.0, color='black', linestyle='--', linewidth=2, label='Equal coupling')
ax2.axvline(np.mean(ratios_all), color='red', linestyle='-', linewidth=2, label=f'Mean={np.mean(ratios_all):.2f}')
ax2.set_xlabel('Coupling Ratio (phi:1 / 2:1)')
ax2.set_ylabel('Count')
ax2.set_title('B. Distribution of Coupling Ratios')
ax2.legend()

ax3 = axes[1, 0]
ax3.scatter(conv_all, ratios_all, c='steelblue', s=100, alpha=0.7)
ax3.axhline(1.0, color='black', linestyle='--', alpha=0.5)
ax3.set_xlabel('8 Hz Convergence (%)')
ax3.set_ylabel('Coupling Ratio (phi/2:1)')
ax3.set_title('C. 8-8 Convergence (infinity state) vs Coupling Mode')

ax3.text(0.05, 0.95, f'r = {r:.3f}, p = {p:.4f}', transform=ax3.transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

ax4 = axes[1, 1]
ax4.axis('off')

interpretation = f"""
CROSS-FREQUENCY COUPLING ANALYSIS
------------------------------------------

KEY FINDINGS:

1. PHASE LOCKING VALUES
   * 2:1 Harmonic (thinking mode): {np.mean(plv_2_1_all):.3f}
   * phi:1 Golden (receptive mode): {np.mean(plv_phi_all):.3f}
   
2. DOMINANT COUPLING
   * {phi_dominant}/{len(ratios_all)} subjects show phi-dominance
   * {harmonic_dominant}/{len(ratios_all)} subjects show 2:1-dominance
   
3. 8-8 Hz CONVERGENCE (infinity state)
   * Average: {np.mean(conv_all):.1f}% of recording time
   * Correlation with phi-coupling: r = {r:.3f}

INTERPRETATION:
{"TENDENCY TOWARD phi-COUPLING" if np.mean(ratios_all) > 1 else "TENDENCY TOWARD 2:1-COUPLING"}

The {"positive" if r > 0 else "negative"} correlation (r={r:.3f}) between 
8 Hz convergence and phi-coupling suggests that the 
"infinity state" (theta-alpha boundary fusion) 
{"IS" if abs(r) > 0.3 else "may be"} associated with 
the golden ratio decoupling mode.

phi = {PHI:.4f}
"""
ax4.text(0.05, 0.95, interpretation, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('cross_frequency_coupling.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: cross_frequency_coupling.png")
