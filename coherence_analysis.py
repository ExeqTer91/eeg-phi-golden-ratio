import numpy as np
from scipy import signal
from scipy.stats import pearsonr, ttest_ind
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf

mne.set_log_level('ERROR')

PHI = (1 + np.sqrt(5)) / 2
print("="*70)
print("COHERENCE ANALYSIS - Global Brain Synchronization")
print("Measuring inter-regional connectivity in theta and alpha bands")
print("="*70)

BANDS = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'low_alpha': (8, 10),
    'high_alpha': (10, 13),
    'theta_alpha_boundary': (7, 9)
}

def compute_coherence_matrix(raw, band):
    data = raw.get_data()
    n_channels = data.shape[0]
    sfreq = raw.info['sfreq']
    
    coherence_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            f, Pxy = signal.csd(data[i], data[j], sfreq, nperseg=1024)
            f, Pxx = signal.welch(data[i], sfreq, nperseg=1024)
            f, Pyy = signal.welch(data[j], sfreq, nperseg=1024)
            
            coh = np.abs(Pxy)**2 / (Pxx * Pyy + 1e-10)
            
            mask = (f >= band[0]) & (f <= band[1])
            band_coh = np.mean(coh[mask]) if np.any(mask) else 0
            
            coherence_matrix[i, j] = band_coh
            coherence_matrix[j, i] = band_coh
    
    return coherence_matrix

def compute_global_coherence(raw, band):
    coh_matrix = compute_coherence_matrix(raw, band)
    upper_tri = coh_matrix[np.triu_indices_from(coh_matrix, k=1)]
    return {
        'mean': np.mean(upper_tri),
        'std': np.std(upper_tri),
        'max': np.max(upper_tri),
        'min': np.min(upper_tri)
    }

def compute_frontal_posterior_coherence(raw, band):
    ch_names = [ch.upper() for ch in raw.ch_names]
    
    frontal_idx = [i for i, ch in enumerate(ch_names) if 'F' in ch and 'P' not in ch]
    posterior_idx = [i for i, ch in enumerate(ch_names) if 'P' in ch or 'O' in ch]
    
    if not frontal_idx or not posterior_idx:
        return np.nan
    
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    
    coherences = []
    for f_idx in frontal_idx[:4]:
        for p_idx in posterior_idx[:4]:
            f, Pxy = signal.csd(data[f_idx], data[p_idx], sfreq, nperseg=1024)
            f, Pxx = signal.welch(data[f_idx], sfreq, nperseg=1024)
            f, Pyy = signal.welch(data[p_idx], sfreq, nperseg=1024)
            
            coh = np.abs(Pxy)**2 / (Pxx * Pyy + 1e-10)
            mask = (f >= band[0]) & (f <= band[1])
            coherences.append(np.mean(coh[mask]) if np.any(mask) else 0)
    
    return np.mean(coherences) if coherences else np.nan

def compute_hemispheric_coherence(raw, band):
    ch_names = [ch.upper() for ch in raw.ch_names]
    
    left_idx = [i for i, ch in enumerate(ch_names) if any(c in ch for c in ['1', '3', '5', '7'])]
    right_idx = [i for i, ch in enumerate(ch_names) if any(c in ch for c in ['2', '4', '6', '8'])]
    
    if not left_idx or not right_idx:
        return np.nan
    
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    
    coherences = []
    for l_idx in left_idx[:4]:
        for r_idx in right_idx[:4]:
            f, Pxy = signal.csd(data[l_idx], data[r_idx], sfreq, nperseg=1024)
            f, Pxx = signal.welch(data[l_idx], sfreq, nperseg=1024)
            f, Pyy = signal.welch(data[r_idx], sfreq, nperseg=1024)
            
            coh = np.abs(Pxy)**2 / (Pxx * Pyy + 1e-10)
            mask = (f >= band[0]) & (f <= band[1])
            coherences.append(np.mean(coh[mask]) if np.any(mask) else 0)
    
    return np.mean(coherences) if coherences else np.nan

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

def pci(ratio):
    if np.isnan(ratio) or ratio <= 0:
        return np.nan
    d_phi = abs(ratio - PHI)
    d_2 = abs(ratio - 2.0)
    return (d_2 - d_phi) / (d_2 + d_phi) if (d_2 + d_phi) > 0 else 0

print("\nAnalyzing coherence across subjects...")

all_results = []

for subj in range(1, 31):
    try:
        raw = read_raw_edf(eegbci.load_data(subj, [1], update_path=True, verbose=False)[0], 
                          preload=True, verbose=False)
        raw.filter(1, 45, verbose=False)
        sfreq = raw.info['sfreq']
        eeg_mean = np.mean(raw.get_data(), axis=0)
        
        f, p = signal.welch(eeg_mean, sfreq, nperseg=1024)
        t_mask = (f >= 4) & (f <= 8)
        a_mask = (f >= 8) & (f <= 13)
        f_theta = np.sum(f[t_mask] * p[t_mask]) / np.sum(p[t_mask]) if np.sum(p[t_mask]) > 0 else np.nan
        f_alpha = np.sum(f[a_mask] * p[a_mask]) / np.sum(p[a_mask]) if np.sum(p[a_mask]) > 0 else np.nan
        ratio = f_alpha / f_theta if f_theta > 0 else np.nan
        
        convergence = compute_8hz_convergence(eeg_mean, sfreq)
        
        theta_coh = compute_global_coherence(raw, BANDS['theta'])
        alpha_coh = compute_global_coherence(raw, BANDS['alpha'])
        boundary_coh = compute_global_coherence(raw, BANDS['theta_alpha_boundary'])
        
        fp_theta = compute_frontal_posterior_coherence(raw, BANDS['theta'])
        fp_alpha = compute_frontal_posterior_coherence(raw, BANDS['alpha'])
        hem_theta = compute_hemispheric_coherence(raw, BANDS['theta'])
        hem_alpha = compute_hemispheric_coherence(raw, BANDS['alpha'])
        
        result = {
            'subject': subj,
            'ratio': ratio,
            'pci': pci(ratio),
            'convergence': convergence,
            'f_theta': f_theta,
            'f_alpha': f_alpha,
            'theta_coherence': theta_coh['mean'],
            'alpha_coherence': alpha_coh['mean'],
            'boundary_coherence': boundary_coh['mean'],
            'frontal_posterior_theta': fp_theta,
            'frontal_posterior_alpha': fp_alpha,
            'hemispheric_theta': hem_theta,
            'hemispheric_alpha': hem_alpha,
        }
        
        all_results.append(result)
        print(f"S{subj:02d}: PCI={result['pci']:+.3f}, Conv={convergence:.1f}%, "
              f"theta-coh={theta_coh['mean']:.3f}, alpha-coh={alpha_coh['mean']:.3f}, "
              f"boundary-coh={boundary_coh['mean']:.3f}")
        
    except Exception as e:
        print(f"S{subj:02d}: Error - {e}")
        continue

print(f"\nSuccessfully analyzed {len(all_results)} subjects")

print("\n" + "="*70)
print("CORRELATION ANALYSIS: What predicts infinity-state access?")
print("="*70)

pcis = [r['pci'] for r in all_results]
convs = [r['convergence'] for r in all_results]
theta_cohs = [r['theta_coherence'] for r in all_results]
alpha_cohs = [r['alpha_coherence'] for r in all_results]
boundary_cohs = [r['boundary_coherence'] for r in all_results]
fp_thetas = [r['frontal_posterior_theta'] for r in all_results]
fp_alphas = [r['frontal_posterior_alpha'] for r in all_results]

correlations = [
    ('PCI', pcis, convs, '8Hz Convergence'),
    ('Theta Coherence', theta_cohs, convs, '8Hz Convergence'),
    ('Alpha Coherence', alpha_cohs, convs, '8Hz Convergence'),
    ('Boundary Coherence (7-9Hz)', boundary_cohs, convs, '8Hz Convergence'),
    ('F-P Theta Coh', fp_thetas, convs, '8Hz Convergence'),
    ('F-P Alpha Coh', fp_alphas, convs, '8Hz Convergence'),
    ('Theta Coherence', theta_cohs, pcis, 'PCI'),
    ('Alpha Coherence', alpha_cohs, pcis, 'PCI'),
    ('Boundary Coherence', boundary_cohs, pcis, 'PCI'),
]

print(f"\n{'Predictor':<30} {'Outcome':<20} {'r':>8} {'p':>10} {'Sig':>5}")
print("-"*75)

significant_findings = []

for pred_name, pred_vals, out_vals, out_name in correlations:
    valid = [(p, o) for p, o in zip(pred_vals, out_vals) if not np.isnan(p) and not np.isnan(o)]
    if len(valid) >= 5:
        pred_clean = [v[0] for v in valid]
        out_clean = [v[1] for v in valid]
        r, p = pearsonr(pred_clean, out_clean)
        sig = "**" if p < 0.05 else ("*" if p < 0.1 else "")
        print(f"{pred_name:<30} {out_name:<20} {r:>8.3f} {p:>10.4f} {sig}")
        
        if p < 0.1:
            significant_findings.append({
                'predictor': pred_name,
                'outcome': out_name,
                'r': r,
                'p': p
            })

print("\n" + "="*70)
print("HIGH vs LOW INFINITY-ACCESSORS: Coherence Differences")
print("="*70)

median_conv = np.median(convs)
high_conv = [r for r in all_results if r['convergence'] > median_conv]
low_conv = [r for r in all_results if r['convergence'] <= median_conv]

coherence_metrics = [
    ('theta_coherence', 'Global Theta Coherence'),
    ('alpha_coherence', 'Global Alpha Coherence'),
    ('boundary_coherence', 'Boundary Coherence (7-9Hz)'),
    ('frontal_posterior_theta', 'Frontal-Posterior Theta'),
    ('frontal_posterior_alpha', 'Frontal-Posterior Alpha'),
    ('hemispheric_theta', 'Inter-hemispheric Theta'),
    ('hemispheric_alpha', 'Inter-hemispheric Alpha'),
]

print(f"\nMedian convergence: {median_conv:.1f}%")
print(f"HIGH group: N={len(high_conv)}, LOW group: N={len(low_conv)}")
print(f"\n{'Metric':<30} {'HIGH':>15} {'LOW':>15} {'t':>8} {'p':>10} {'Sig':>5}")
print("-"*85)

for metric, label in coherence_metrics:
    high_vals = [r[metric] for r in high_conv if not np.isnan(r[metric])]
    low_vals = [r[metric] for r in low_conv if not np.isnan(r[metric])]
    
    if len(high_vals) >= 2 and len(low_vals) >= 2:
        t, p = ttest_ind(high_vals, low_vals)
        sig = "**" if p < 0.05 else ("*" if p < 0.1 else "")
        print(f"{label:<30} {np.mean(high_vals):>7.3f}+/-{np.std(high_vals):<5.3f} "
              f"{np.mean(low_vals):>7.3f}+/-{np.std(low_vals):<5.3f} "
              f"{t:>8.2f} {p:>10.4f} {sig}")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

ax1 = axes[0, 0]
valid_pairs = [(b, c) for b, c in zip(boundary_cohs, convs) if not np.isnan(b)]
ax1.scatter([v[0] for v in valid_pairs], [v[1] for v in valid_pairs], c='purple', s=80, alpha=0.7)
r, p = pearsonr([v[0] for v in valid_pairs], [v[1] for v in valid_pairs])
ax1.set_xlabel('Boundary Coherence (7-9 Hz)')
ax1.set_ylabel('8 Hz Convergence (%)')
ax1.set_title(f'A. Boundary Coherence vs Infinity-State\nr={r:.3f}, p={p:.4f}')

ax2 = axes[0, 1]
valid_pairs = [(t, c) for t, c in zip(theta_cohs, convs) if not np.isnan(t)]
ax2.scatter([v[0] for v in valid_pairs], [v[1] for v in valid_pairs], c='green', s=80, alpha=0.7)
r, p = pearsonr([v[0] for v in valid_pairs], [v[1] for v in valid_pairs])
ax2.set_xlabel('Global Theta Coherence')
ax2.set_ylabel('8 Hz Convergence (%)')
ax2.set_title(f'B. Theta Coherence vs Infinity-State\nr={r:.3f}, p={p:.4f}')

ax3 = axes[0, 2]
valid_pairs = [(a, c) for a, c in zip(alpha_cohs, convs) if not np.isnan(a)]
ax3.scatter([v[0] for v in valid_pairs], [v[1] for v in valid_pairs], c='blue', s=80, alpha=0.7)
r, p = pearsonr([v[0] for v in valid_pairs], [v[1] for v in valid_pairs])
ax3.set_xlabel('Global Alpha Coherence')
ax3.set_ylabel('8 Hz Convergence (%)')
ax3.set_title(f'C. Alpha Coherence vs Infinity-State\nr={r:.3f}, p={p:.4f}')

ax4 = axes[1, 0]
metrics = ['Theta\nCoh', 'Alpha\nCoh', 'Bound.\nCoh', 'F-P\nTheta', 'F-P\nAlpha']
high_means = [np.mean([r['theta_coherence'] for r in high_conv]),
              np.mean([r['alpha_coherence'] for r in high_conv]),
              np.mean([r['boundary_coherence'] for r in high_conv]),
              np.nanmean([r['frontal_posterior_theta'] for r in high_conv]),
              np.nanmean([r['frontal_posterior_alpha'] for r in high_conv])]
low_means = [np.mean([r['theta_coherence'] for r in low_conv]),
             np.mean([r['alpha_coherence'] for r in low_conv]),
             np.mean([r['boundary_coherence'] for r in low_conv]),
             np.nanmean([r['frontal_posterior_theta'] for r in low_conv]),
             np.nanmean([r['frontal_posterior_alpha'] for r in low_conv])]

x = np.arange(len(metrics))
width = 0.35
ax4.bar(x - width/2, high_means, width, label='HIGH', color='gold')
ax4.bar(x + width/2, low_means, width, label='LOW', color='gray')
ax4.set_xticks(x)
ax4.set_xticklabels(metrics)
ax4.set_ylabel('Coherence')
ax4.set_title('D. Coherence: HIGH vs LOW Infinity-Accessors')
ax4.legend()

ax5 = axes[1, 1]
valid_pairs = [(b, pc) for b, pc in zip(boundary_cohs, pcis) if not np.isnan(b) and not np.isnan(pc)]
ax5.scatter([v[0] for v in valid_pairs], [v[1] for v in valid_pairs], c='orange', s=80, alpha=0.7)
r, p = pearsonr([v[0] for v in valid_pairs], [v[1] for v in valid_pairs])
ax5.axhline(0, color='black', linestyle='--', alpha=0.5)
ax5.set_xlabel('Boundary Coherence (7-9 Hz)')
ax5.set_ylabel('PCI (phi-Coupling Index)')
ax5.set_title(f'E. Boundary Coherence vs phi-Coupling\nr={r:.3f}, p={p:.4f}')

ax6 = axes[1, 2]
ax6.axis('off')

summary = f"""
COHERENCE ANALYSIS SUMMARY
------------------------------------------

N = {len(all_results)} subjects analyzed

KEY CORRELATIONS WITH INFINITY-STATE:
"""
for f in significant_findings[:5]:
    summary += f"\n* {f['predictor']} -> {f['outcome']}: r={f['r']:.3f}, p={f['p']:.4f}"

summary += f"""

HIGH vs LOW INFINITY-ACCESSORS:
* HIGH group (>{median_conv:.1f}%): N={len(high_conv)}
* LOW group (<={median_conv:.1f}%): N={len(low_conv)}

INTERPRETATION:
Coherence in the theta-alpha boundary zone (7-9 Hz)
may be a neural marker of infinity-state access.

phi = {PHI:.4f}
"""

ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('coherence_analysis.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: coherence_analysis.png")
