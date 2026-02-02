import numpy as np
from scipy import signal
from scipy.stats import ttest_ind, pearsonr, spearmanr
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf

mne.set_log_level('ERROR')

PHI = (1 + np.sqrt(5)) / 2
print("="*70)
print("DEEP ANALYSIS: HIGH vs LOW INFINITY-STATE SUBJECTS")
print("What makes some people natural 'infinity accessors'?")
print("="*70)

BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

def full_spectral_analysis(eeg, sfreq):
    freqs, psd = signal.welch(eeg, sfreq, nperseg=min(1024, len(eeg)//2))
    
    results = {}
    total_power = np.sum(psd[(freqs >= 1) & (freqs <= 45)])
    
    for band_name, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs <= high)
        band_power = np.sum(psd[mask])
        
        results[f'{band_name}_power'] = band_power
        results[f'{band_name}_relative'] = band_power / total_power if total_power > 0 else 0
        
        if np.any(mask) and np.sum(psd[mask]) > 0:
            band_freqs = freqs[mask]
            band_psd = psd[mask]
            results[f'{band_name}_peak'] = band_freqs[np.argmax(band_psd)]
            results[f'{band_name}_centroid'] = np.sum(band_freqs * band_psd) / np.sum(band_psd)
        else:
            results[f'{band_name}_peak'] = np.nan
            results[f'{band_name}_centroid'] = np.nan
    
    results['theta_alpha_ratio'] = results['alpha_centroid'] / results['theta_centroid'] if results['theta_centroid'] > 0 else np.nan
    results['theta_alpha_power_ratio'] = results['theta_power'] / results['alpha_power'] if results['alpha_power'] > 0 else np.nan
    
    log_freqs = np.log10(freqs[(freqs >= 2) & (freqs <= 40)])
    log_psd = np.log10(psd[(freqs >= 2) & (freqs <= 40)])
    if len(log_freqs) > 2:
        slope, _ = np.polyfit(log_freqs, log_psd, 1)
        results['spectral_slope'] = slope
    else:
        results['spectral_slope'] = np.nan
    
    return results

def compute_8hz_convergence(eeg, sfreq):
    window_sec = 2
    window_samples = int(window_sec * sfreq)
    step = window_samples // 2
    
    convergence_count = 0
    total_windows = 0
    convergence_strengths = []
    
    for start in range(0, len(eeg) - window_samples, step):
        segment = eeg[start:start + window_samples]
        freqs, psd = signal.welch(segment, sfreq, nperseg=min(512, len(segment)))
        
        theta_mask = (freqs >= 4) & (freqs <= 8)
        if np.any(theta_mask):
            f_theta = freqs[theta_mask][np.argmax(psd[theta_mask])]
        else:
            continue
            
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        if np.any(alpha_mask):
            f_alpha = freqs[alpha_mask][np.argmax(psd[alpha_mask])]
        else:
            continue
        
        total_windows += 1
        
        distance = abs(f_theta - f_alpha)
        near_8 = (7 <= f_theta <= 8.5) and (7.5 <= f_alpha <= 9)
        
        if near_8 and distance < 1.5:
            convergence_count += 1
            convergence_strengths.append(1 - distance/1.5)
    
    convergence_pct = 100 * convergence_count / total_windows if total_windows > 0 else 0
    mean_strength = np.mean(convergence_strengths) if convergence_strengths else 0
    
    return {
        'convergence_pct': convergence_pct,
        'convergence_strength': mean_strength,
        'total_windows': total_windows
    }

def compute_phase_metrics(eeg, sfreq):
    b_t, a_t = signal.butter(4, [4/(sfreq/2), 8/(sfreq/2)], 'band')
    b_a, a_a = signal.butter(4, [8/(sfreq/2), 13/(sfreq/2)], 'band')
    
    theta = signal.filtfilt(b_t, a_t, eeg)
    alpha = signal.filtfilt(b_a, a_a, eeg)
    
    theta_phase = np.angle(signal.hilbert(theta))
    alpha_phase = np.angle(signal.hilbert(alpha))
    
    plv_2_1 = np.abs(np.mean(np.exp(1j * (alpha_phase - 2 * theta_phase))))
    plv_phi = np.abs(np.mean(np.exp(1j * (alpha_phase - PHI * theta_phase))))
    plv_1_1 = np.abs(np.mean(np.exp(1j * (alpha_phase - theta_phase))))
    
    phase_diff = alpha_phase - theta_phase
    phase_stability = 1 - np.std(np.diff(phase_diff)) / np.pi
    
    return {
        'plv_2_1': plv_2_1,
        'plv_phi': plv_phi,
        'plv_1_1': plv_1_1,
        'phase_stability': phase_stability,
        'coupling_ratio': plv_phi / plv_2_1 if plv_2_1 > 0 else np.nan
    }

print("\nAnalyzing 30 subjects in detail...")

all_subjects = []

for subj in range(1, 31):
    try:
        raw = read_raw_edf(eegbci.load_data(subj, [1], update_path=True, verbose=False)[0], 
                          preload=True, verbose=False)
        raw.filter(1, 45, verbose=False)
        sfreq = raw.info['sfreq']
        eeg = np.mean(raw.get_data(), axis=0)
        
        spectral = full_spectral_analysis(eeg, sfreq)
        convergence = compute_8hz_convergence(eeg, sfreq)
        phase = compute_phase_metrics(eeg, sfreq)
        
        subject_data = {
            'subject': subj,
            **spectral,
            **convergence,
            **phase
        }
        all_subjects.append(subject_data)
        
    except Exception as e:
        continue

print(f"Successfully analyzed {len(all_subjects)} subjects")

convergences = [s['convergence_pct'] for s in all_subjects]
median_conv = np.median(convergences)

high_conv = [s for s in all_subjects if s['convergence_pct'] > median_conv]
low_conv = [s for s in all_subjects if s['convergence_pct'] <= median_conv]

print(f"\n" + "="*70)
print(f"GROUP COMPARISON: HIGH vs LOW INFINITY-CONVERGENCE")
print(f"="*70)
print(f"Median convergence: {median_conv:.1f}%")
print(f"HIGH group (>{median_conv:.1f}%): N={len(high_conv)}")
print(f"LOW group (<={median_conv:.1f}%): N={len(low_conv)}")

comparison_metrics = [
    ('theta_relative', 'Theta Relative Power'),
    ('alpha_relative', 'Alpha Relative Power'),
    ('beta_relative', 'Beta Relative Power'),
    ('theta_alpha_ratio', 'Theta-Alpha Frequency Ratio'),
    ('theta_alpha_power_ratio', 'Theta/Alpha Power Ratio'),
    ('spectral_slope', 'Spectral Slope (1/f)'),
    ('plv_2_1', 'PLV 2:1 (harmonic)'),
    ('plv_phi', 'PLV phi:1 (golden)'),
    ('plv_1_1', 'PLV 1:1 (fusion)'),
    ('coupling_ratio', 'Coupling Ratio (phi/2:1)'),
    ('phase_stability', 'Phase Stability'),
    ('theta_centroid', 'Theta Centroid (Hz)'),
    ('alpha_centroid', 'Alpha Centroid (Hz)'),
]

print(f"\n{'Metric':<30} {'HIGH':<20} {'LOW':<20} {'t-stat':>8} {'p-value':>10} {'Sig':>5}")
print("-"*95)

significant_findings = []

for metric, label in comparison_metrics:
    high_vals = [s[metric] for s in high_conv if not np.isnan(s[metric])]
    low_vals = [s[metric] for s in low_conv if not np.isnan(s[metric])]
    
    if len(high_vals) >= 2 and len(low_vals) >= 2:
        t_stat, p_val = ttest_ind(high_vals, low_vals)
        sig = "**" if p_val < 0.05 else ("*" if p_val < 0.1 else "")
        
        print(f"{label:<30} {np.mean(high_vals):>8.3f}+/-{np.std(high_vals):<7.3f} "
              f"{np.mean(low_vals):>8.3f}+/-{np.std(low_vals):<7.3f} "
              f"{t_stat:>8.2f} {p_val:>10.4f} {sig:>5}")
        
        if p_val < 0.1:
            significant_findings.append({
                'metric': label,
                'high_mean': np.mean(high_vals),
                'low_mean': np.mean(low_vals),
                'p_value': p_val,
                'direction': 'HIGH > LOW' if np.mean(high_vals) > np.mean(low_vals) else 'HIGH < LOW'
            })

print(f"\n" + "="*70)
print("TOP 5 INFINITY ACCESSORS - Individual Profiles")
print("="*70)

top_5 = sorted(all_subjects, key=lambda x: x['convergence_pct'], reverse=True)[:5]

for i, s in enumerate(top_5):
    print(f"\n#{i+1} Subject {s['subject']}: {s['convergence_pct']:.1f}% convergence")
    print(f"    Theta-Alpha ratio: {s['theta_alpha_ratio']:.3f} (phi={PHI:.3f})")
    print(f"    PLV 1:1 (fusion): {s['plv_1_1']:.3f}")
    print(f"    PLV phi:1: {s['plv_phi']:.3f}")
    print(f"    Coupling ratio: {s['coupling_ratio']:.3f}")
    print(f"    Theta centroid: {s['theta_centroid']:.2f} Hz")
    print(f"    Alpha centroid: {s['alpha_centroid']:.2f} Hz")
    print(f"    Phase stability: {s['phase_stability']:.3f}")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

ax1 = axes[0, 0]
ax1.hist(convergences, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(median_conv, color='red', linestyle='--', linewidth=2, label=f'Median={median_conv:.1f}%')
ax1.set_xlabel('8 Hz Convergence (%)')
ax1.set_ylabel('Count')
ax1.set_title('A. Distribution of Infinity-State Access')
ax1.legend()

ax2 = axes[0, 1]
metrics_to_plot = ['plv_1_1', 'plv_phi', 'coupling_ratio']
labels = ['PLV 1:1\n(fusion)', 'PLV phi:1\n(golden)', 'Coupling\nRatio']
x = np.arange(len(metrics_to_plot))
width = 0.35

high_means = [np.mean([s[m] for s in high_conv]) for m in metrics_to_plot]
low_means = [np.mean([s[m] for s in low_conv]) for m in metrics_to_plot]
high_stds = [np.std([s[m] for s in high_conv]) for m in metrics_to_plot]
low_stds = [np.std([s[m] for s in low_conv]) for m in metrics_to_plot]

ax2.bar(x - width/2, high_means, width, yerr=high_stds, label='HIGH', color='gold', capsize=5)
ax2.bar(x + width/2, low_means, width, yerr=low_stds, label='LOW', color='gray', capsize=5)
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_ylabel('Value')
ax2.set_title('B. Phase Coupling: HIGH vs LOW Infinity-Accessors')
ax2.legend()

ax3 = axes[0, 2]
for s in high_conv:
    ax3.scatter(s['theta_centroid'], s['alpha_centroid'], c='gold', s=100, alpha=0.7, label='HIGH' if s == high_conv[0] else '')
for s in low_conv:
    ax3.scatter(s['theta_centroid'], s['alpha_centroid'], c='gray', s=100, alpha=0.7, label='LOW' if s == low_conv[0] else '')
ax3.plot([5, 8], [8, 8], 'r--', linewidth=2, label='8 Hz line')
ax3.plot([8, 8], [8, 13], 'r--', linewidth=2)
ax3.set_xlabel('Theta Centroid (Hz)')
ax3.set_ylabel('Alpha Centroid (Hz)')
ax3.set_title('C. Theta-Alpha Frequency Space')
ax3.legend()

ax4 = axes[1, 0]
conv_vals = [s['convergence_pct'] for s in all_subjects]
plv11_vals = [s['plv_1_1'] for s in all_subjects]
ax4.scatter(conv_vals, plv11_vals, c='purple', s=80, alpha=0.7)
r, p = pearsonr(conv_vals, plv11_vals)
ax4.set_xlabel('8 Hz Convergence (%)')
ax4.set_ylabel('PLV 1:1 (Fusion)')
ax4.set_title(f'D. Infinity-State vs 1:1 Phase Fusion (r={r:.3f}, p={p:.3f})')

ax5 = axes[1, 1]
high_slopes = [s['spectral_slope'] for s in high_conv]
low_slopes = [s['spectral_slope'] for s in low_conv]
ax5.boxplot([high_slopes, low_slopes], labels=['HIGH', 'LOW'])
ax5.set_ylabel('Spectral Slope (1/f)')
ax5.set_title('E. Neural Complexity (1/f Slope)')
t, p = ttest_ind(high_slopes, low_slopes)
ax5.text(0.5, 0.95, f't={t:.2f}, p={p:.3f}', transform=ax5.transAxes, ha='center')

ax6 = axes[1, 2]
ax6.axis('off')

summary_text = f"""
HIGH vs LOW INFINITY-ACCESSORS ANALYSIS
------------------------------------------

N = {len(all_subjects)} subjects
Median 8 Hz convergence: {median_conv:.1f}%
HIGH group: N={len(high_conv)} (>{median_conv:.1f}%)
LOW group: N={len(low_conv)} (<={median_conv:.1f}%)

SIGNIFICANT FINDINGS (p < 0.1):
"""
for f in significant_findings:
    summary_text += f"\n* {f['metric']}: {f['direction']} (p={f['p_value']:.3f})"

if not significant_findings:
    summary_text += "\nNo significant differences at p < 0.1"

summary_text += f"""

TOP INFINITY ACCESSOR:
Subject {top_5[0]['subject']}: {top_5[0]['convergence_pct']:.1f}% convergence
  Theta-Alpha ratio: {top_5[0]['theta_alpha_ratio']:.3f}
  PLV 1:1: {top_5[0]['plv_1_1']:.3f}
  
phi = {PHI:.4f}
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('infinity_accessors_analysis.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: infinity_accessors_analysis.png")

print("\n" + "="*70)
print("CORRELATION ANALYSIS")
print("="*70)

correlations = [
    ('convergence_pct', 'plv_1_1', 'Convergence vs PLV 1:1'),
    ('convergence_pct', 'plv_phi', 'Convergence vs PLV phi'),
    ('convergence_pct', 'coupling_ratio', 'Convergence vs Coupling Ratio'),
    ('convergence_pct', 'theta_alpha_ratio', 'Convergence vs Theta-Alpha Ratio'),
    ('convergence_pct', 'spectral_slope', 'Convergence vs Spectral Slope'),
    ('plv_1_1', 'plv_phi', 'PLV 1:1 vs PLV phi'),
]

for var1, var2, label in correlations:
    vals1 = [s[var1] for s in all_subjects if not np.isnan(s[var1]) and not np.isnan(s[var2])]
    vals2 = [s[var2] for s in all_subjects if not np.isnan(s[var1]) and not np.isnan(s[var2])]
    if len(vals1) >= 3:
        r, p = pearsonr(vals1, vals2)
        sig = "**" if p < 0.05 else ("*" if p < 0.1 else "")
        print(f"{label:<40} r = {r:+.3f}, p = {p:.4f} {sig}")
