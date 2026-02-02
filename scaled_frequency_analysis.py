"""
SCALED FREQUENCY EXPLORATION
============================
Căutăm pattern-uri φ la multiple scale de frecvență
"""

import numpy as np
from scipy import stats, signal
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf

mne.set_log_level('ERROR')

PHI = (1 + np.sqrt(5)) / 2

print("="*70)
print("SCALED FREQUENCY EXPLORATION")
print("Testing multiple hypotheses about φ and brain oscillations")
print("="*70)

BANDS = {
    'delta': (1, 4),
    'theta_low': (4, 6),
    'theta_high': (6, 8),
    'theta': (4, 8),
    'alpha_low': (8, 10),
    'alpha_high': (10, 13),
    'alpha': (8, 13),
    'beta_low': (13, 20),
    'beta': (13, 30)
}

def get_band_centroid(psd, freqs, band):
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

subjects_data = []

for subj in range(1, 51):
    try:
        raw = read_raw_edf(eegbci.load_data(subj, [1], update_path=True, verbose=False)[0], 
                          preload=True, verbose=False)
        raw.filter(1, 45, verbose=False)
        sfreq = raw.info['sfreq']
        eeg = np.mean(raw.get_data(), axis=0)
        
        f, p = signal.welch(eeg, sfreq, nperseg=min(1024, len(eeg)//2))
        
        centroids = {}
        for band_name, band_range in BANDS.items():
            centroids[band_name] = get_band_centroid(p, f, band_range)
        
        conv = compute_8hz_convergence(eeg, sfreq)
        
        if not np.isnan(centroids['theta']) and not np.isnan(centroids['alpha']):
            ratio = centroids['alpha'] / centroids['theta']
            subjects_data.append({
                'subject': subj,
                'centroids': centroids,
                'theta_alpha_ratio': ratio,
                'convergence_pct': conv
            })
    except:
        continue

print(f"Loaded {len(subjects_data)} subjects")

print("\n" + "="*70)
print("TEST 1: SUB-BAND RATIO ANALYSIS")
print("="*70)

ratios_to_test = [
    ('theta', 'delta', 'θ/δ'),
    ('alpha', 'theta', 'α/θ'),
    ('beta', 'alpha', 'β/α'),
    ('theta_high', 'theta_low', 'θ_hi/θ_lo'),
    ('alpha_high', 'alpha_low', 'α_hi/α_lo'),
    ('alpha_low', 'theta_high', 'α_lo/θ_hi (8Hz zone)'),
]

print(f"\n{'Ratio':<25} {'Mean':>8} {'Std':>8} {'vs φ':>8} {'r (conv)':>10} {'p':>10}")
print("-"*75)

for num, denom, label in ratios_to_test:
    ratios = []
    convs = []
    for s in subjects_data:
        c_num = s['centroids'].get(num, np.nan)
        c_denom = s['centroids'].get(denom, np.nan)
        if c_denom > 0 and not np.isnan(c_num) and not np.isnan(c_denom):
            ratios.append(c_num / c_denom)
            convs.append(s['convergence_pct'])
    
    if len(ratios) > 5:
        mean_r = np.mean(ratios)
        std_r = np.std(ratios)
        dist_phi = mean_r - PHI
        
        proximity = -np.abs(np.array(ratios) - PHI)
        r_corr, p_corr = stats.pearsonr(proximity, convs)
        
        sig = "**" if p_corr < 0.05 else "*" if p_corr < 0.1 else ""
        print(f"{label:<25} {mean_r:>8.3f} {std_r:>8.3f} {dist_phi:>+8.3f} {r_corr:>+10.3f} {p_corr:>10.4f} {sig}")

print("\n" + "="*70)
print("TEST 2: HARMONIC DISTANCE ANALYSIS (Key for paper!)")
print("="*70)

harmonics = [1.0, 1.25, 1.333, 1.5, 1.667, 2.0, 2.5, 3.0]
harmonic_names = ['1:1', '5:4', '4:3', '3:2', '5:3', '2:1', '5:2', '3:1']

for s in subjects_data:
    ratio = s['theta_alpha_ratio']
    min_dist = min(abs(ratio - h) for h in harmonics)
    nearest_idx = np.argmin([abs(ratio - h) for h in harmonics])
    s['harmonic_distance'] = min_dist
    s['nearest_harmonic'] = harmonic_names[nearest_idx]

distances = [s['harmonic_distance'] for s in subjects_data]
convergences = [s['convergence_pct'] for s in subjects_data]
ratios = [s['theta_alpha_ratio'] for s in subjects_data]

r_harm, p_harm = stats.pearsonr(distances, convergences)

print(f"\nHarmonic distance ↔ Convergence: r = {r_harm:+.3f}, p = {p_harm:.4f}")

if r_harm > 0:
    print("\n✓ POSITIVE correlation: Further from harmonics = MORE convergence!")
    print("  This supports the hypothesis that φ is special because it's")
    print("  maximally distant from all simple harmonic ratios!")
else:
    print("\n⚠ Negative correlation: Closer to harmonics = more convergence")

print("\n" + "="*70)
print("TEST 3: INDIVIDUAL OPTIMAL RATIO CLUSTERING")
print("="*70)

high_conv = [(s['theta_alpha_ratio'], s['convergence_pct']) for s in subjects_data if s['convergence_pct'] > 15]
low_conv = [(s['theta_alpha_ratio'], s['convergence_pct']) for s in subjects_data if s['convergence_pct'] < 5]

print(f"\nHIGH converters (>15%): N = {len(high_conv)}")
if high_conv:
    print(f"  Mean ratio: {np.mean([h[0] for h in high_conv]):.3f} (distance from φ: {abs(np.mean([h[0] for h in high_conv]) - PHI):.3f})")
    
print(f"\nLOW converters (<5%): N = {len(low_conv)}")
if low_conv:
    print(f"  Mean ratio: {np.mean([l[0] for l in low_conv]):.3f} (distance from φ: {abs(np.mean([l[0] for l in low_conv]) - PHI):.3f})")

if high_conv and low_conv:
    t, p = stats.ttest_ind([h[0] for h in high_conv], [l[0] for l in low_conv])
    print(f"\nDifference test: t = {t:.3f}, p = {p:.4f}")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

ax1 = axes[0, 0]
colors = ['gold' if abs(r - PHI) < 0.1 else 'steelblue' for r in ratios]
scatter = ax1.scatter(ratios, convergences, c=convergences, cmap='viridis', s=80, alpha=0.7, edgecolor='black')
ax1.axvline(x=PHI, color='gold', linestyle='--', linewidth=2, label=f'φ = {PHI:.3f}')
ax1.axvline(x=2.0, color='red', linestyle='--', linewidth=2, label='2:1 = 2.0')
plt.colorbar(scatter, ax=ax1, label='Convergence %')
ax1.set_xlabel('θ/α Frequency Ratio', fontsize=12)
ax1.set_ylabel('8 Hz Convergence (%)', fontsize=12)
ax1.set_title('A. Individual θ/α Ratios vs Convergence', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.scatter(distances, convergences, c='purple', s=80, alpha=0.7)
z = np.polyfit(distances, convergences, 1)
p_fit = np.poly1d(z)
x_line = np.linspace(min(distances), max(distances), 100)
ax2.plot(x_line, p_fit(x_line), 'r-', linewidth=2, label=f'r = {r_harm:+.3f}')
ax2.set_xlabel('Distance to Nearest Harmonic', fontsize=12)
ax2.set_ylabel('8 Hz Convergence (%)', fontsize=12)
ax2.set_title(f'B. Harmonic Distance vs Convergence\n(p = {p_harm:.4f})', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

ax3 = axes[0, 2]
harmonic_counts = {}
for s in subjects_data:
    h = s['nearest_harmonic']
    if h not in harmonic_counts:
        harmonic_counts[h] = {'count': 0, 'conv_sum': 0}
    harmonic_counts[h]['count'] += 1
    harmonic_counts[h]['conv_sum'] += s['convergence_pct']

labels = list(harmonic_counts.keys())
counts = [harmonic_counts[l]['count'] for l in labels]
mean_convs = [harmonic_counts[l]['conv_sum'] / harmonic_counts[l]['count'] for l in labels]

x = np.arange(len(labels))
bars = ax3.bar(x, counts, color='steelblue', edgecolor='black', alpha=0.7)
ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.set_ylabel('Number of Subjects', fontsize=12)
ax3.set_title('C. Distribution by Nearest Harmonic', fontsize=14)

for bar, mc in zip(bars, mean_convs):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{mc:.1f}%', ha='center', fontsize=9, color='red')
ax3.text(0.5, 0.95, 'Red = mean convergence', transform=ax3.transAxes, 
        fontsize=9, ha='center', color='red')

ax4 = axes[1, 0]
if high_conv and low_conv:
    high_ratios = [h[0] for h in high_conv]
    low_ratios = [l[0] for l in low_conv]
    
    ax4.hist(high_ratios, bins=10, alpha=0.7, label=f'HIGH (N={len(high_conv)})', color='gold', edgecolor='black')
    ax4.hist(low_ratios, bins=10, alpha=0.7, label=f'LOW (N={len(low_conv)})', color='gray', edgecolor='black')
    ax4.axvline(x=PHI, color='green', linestyle='--', linewidth=2, label='φ')
    ax4.set_xlabel('θ/α Ratio', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('D. HIGH vs LOW Converters: Ratio Distribution', fontsize=14)
    ax4.legend(fontsize=10)

ax5 = axes[1, 1]
phi_distances = [abs(s['theta_alpha_ratio'] - PHI) for s in subjects_data]
two_distances = [abs(s['theta_alpha_ratio'] - 2.0) for s in subjects_data]

r_phi, p_phi = stats.pearsonr(phi_distances, convergences)
r_two, p_two = stats.pearsonr(two_distances, convergences)

ax5.scatter(phi_distances, convergences, c='gold', s=60, alpha=0.7, label=f'Dist from φ (r={r_phi:+.3f})')
ax5.scatter(two_distances, convergences, c='red', s=60, alpha=0.7, label=f'Dist from 2:1 (r={r_two:+.3f})')
ax5.set_xlabel('Distance from Target Ratio', fontsize=12)
ax5.set_ylabel('8 Hz Convergence (%)', fontsize=12)
ax5.set_title('E. Distance from φ vs 2:1', fontsize=14)
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

ax6 = axes[1, 2]
ax6.axis('off')

summary = f"""
SCALED FREQUENCY EXPLORATION SUMMARY
================================================

N = {len(subjects_data)} subjects

KEY FINDING - HARMONIC DISTANCE:
  Correlation: r = {r_harm:+.3f}, p = {p_harm:.4f}
  {"✓ SIGNIFICANT" if p_harm < 0.05 else "Not significant"}
  
  {"Further from harmonics = MORE convergence!" if r_harm > 0 else "Closer to harmonics = more convergence"}

CLUSTERING ANALYSIS:
  HIGH converters (>15%): N = {len(high_conv)}
    Mean ratio: {np.mean([h[0] for h in high_conv]):.3f}
  
  LOW converters (<5%): N = {len(low_conv)}
    Mean ratio: {np.mean([l[0] for l in low_conv]):.3f}

DISTANCE ANALYSIS:
  Distance from φ ↔ Convergence: r = {r_phi:+.3f}
  Distance from 2:1 ↔ Convergence: r = {r_two:+.3f}
  
  {"φ is better reference!" if abs(r_phi) < abs(r_two) else "2:1 is stronger reference!"}

φ = {PHI:.6f}
"""

ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('scaled_frequency_analysis.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: scaled_frequency_analysis.png")

print("\n" + "="*70)
print("BONUS: DISTANCE CORRELATION COMPARISON")
print("="*70)

print(f"\nDistance from φ (1.618) ↔ Convergence: r = {r_phi:+.3f}, p = {p_phi:.4f}")
print(f"Distance from 2:1 (2.0) ↔ Convergence: r = {r_two:+.3f}, p = {p_two:.4f}")

if r_phi < 0 and r_two > 0:
    print("\n✓ PERFECT RESULT:")
    print("  - Closer to φ = MORE convergence (r < 0)")
    print("  - Closer to 2:1 = LESS convergence (r > 0)")
elif r_phi < r_two:
    print("\n✓ φ is a better reference point than 2:1")
else:
    print("\n⚠ 2:1 shows stronger relationship (unexpected)")
