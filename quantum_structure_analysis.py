"""
OSMOSIS / PHASE TRANSITION / QUANTUM STRUCTURE ANALYSIS
========================================================
Căutăm: faze critice, tranziții, structuri fundamentale
"""

import numpy as np
from scipy import stats, signal
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf

mne.set_log_level('ERROR')

PHI = (1 + np.sqrt(5)) / 2

print("="*70)
print("QUANTUM STRUCTURE ANALYSIS")
print("Looking for phase transitions, quantum levels, and hidden patterns")
print("="*70)

BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
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

def pci(ratio):
    if np.isnan(ratio) or ratio <= 0:
        return np.nan
    d_phi = abs(ratio - PHI)
    d_2 = abs(ratio - 2.0)
    return (d_2 - d_phi) / (d_2 + d_phi) if (d_2 + d_phi) > 0 else 0

print("\nLoading data from 50 subjects...")

all_data = []

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
        ratio = centroids['alpha'] / centroids['theta'] if centroids['theta'] > 0 else np.nan
        
        all_data.append({
            'subject': subj,
            'delta': centroids['delta'],
            'theta': centroids['theta'],
            'alpha': centroids['alpha'],
            'beta': centroids['beta'],
            'ratio': ratio,
            'pci': pci(ratio),
            'convergence': conv
        })
    except:
        continue

print(f"Loaded {len(all_data)} subjects")

print("\n" + "="*70)
print("TEST 1: QUANTUM LEVELS - Discrete frequency states")
print("="*70)

theta_centroids = np.array([d['theta'] for d in all_data])
alpha_centroids = np.array([d['alpha'] for d in all_data])
ratios = np.array([d['ratio'] for d in all_data])
convergences = np.array([d['convergence'] for d in all_data])

x_theta = np.linspace(5, 7, 200)
x_alpha = np.linspace(9, 11, 200)

kde_theta = gaussian_kde(theta_centroids, bw_method=0.05)
kde_alpha = gaussian_kde(alpha_centroids, bw_method=0.05)

y_theta = kde_theta(x_theta)
y_alpha = kde_alpha(x_alpha)

peaks_theta, _ = find_peaks(y_theta, prominence=0.1)
peaks_alpha, _ = find_peaks(y_alpha, prominence=0.1)

print("\nTHETA QUANTUM LEVELS (centroids):")
for p in peaks_theta:
    print(f"  Level: {x_theta[p]:.3f} Hz")

print("\nALPHA QUANTUM LEVELS (centroids):")
for p in peaks_alpha:
    print(f"  Level: {x_alpha[p]:.3f} Hz")

if len(peaks_theta) > 1:
    theta_spacings = np.diff(x_theta[peaks_theta])
    print(f"\nTheta level spacing: {theta_spacings}")

if len(peaks_alpha) > 1:
    alpha_spacings = np.diff(x_alpha[peaks_alpha])
    print(f"Alpha level spacing: {alpha_spacings}")

print("\n" + "="*70)
print("TEST 2: FIBONACCI CASCADE - Cross-band relationships")
print("="*70)

delta_c = np.array([d['delta'] for d in all_data if not np.isnan(d['delta'])])
theta_c = np.array([d['theta'] for d in all_data if not np.isnan(d['theta'])])
alpha_c = np.array([d['alpha'] for d in all_data if not np.isnan(d['alpha'])])
beta_c = np.array([d['beta'] for d in all_data if not np.isnan(d['beta'])])

print("\nMean centroids:")
print(f"  Delta: {np.mean(delta_c):.3f} Hz")
print(f"  Theta: {np.mean(theta_c):.3f} Hz")
print(f"  Alpha: {np.mean(alpha_c):.3f} Hz")
print(f"  Beta: {np.mean(beta_c):.3f} Hz")

print("\nConsecutive band ratios:")
print(f"  θ/δ: {np.mean(theta_c)/np.mean(delta_c):.3f} (vs φ = {PHI:.3f})")
print(f"  α/θ: {np.mean(alpha_c)/np.mean(theta_c):.3f} (vs φ = {PHI:.3f})")
print(f"  β/α: {np.mean(beta_c)/np.mean(alpha_c):.3f} (vs φ = {PHI:.3f})")

print("\n" + "="*70)
print("TEST 3: PHASE TRANSITION - Critical behavior")
print("="*70)

conv_kde = gaussian_kde(convergences, bw_method=0.15)
x_conv = np.linspace(0, 40, 200)
y_conv = conv_kde(x_conv)

peaks_conv, _ = find_peaks(y_conv, prominence=0.01)

print("\nConvergence distribution peaks:")
for p in peaks_conv:
    print(f"  Peak at {x_conv[p]:.1f}%")

if len(peaks_conv) >= 2:
    print("\n✓ BIMODAL distribution detected!")
    print("  Suggests TWO DISTINCT PHASES (non-converged / converged)")
else:
    print("\n⚠ Unimodal distribution")
    print("  Continuous spectrum rather than discrete phases")

variance_windows = []
for threshold in np.linspace(5, 30, 20):
    below = [c for c in convergences if c < threshold]
    above = [c for c in convergences if c >= threshold]
    if len(below) > 2 and len(above) > 2:
        variance_windows.append({
            'threshold': threshold,
            'var_below': np.var(below),
            'var_above': np.var(above),
            'total_var': np.var(below) + np.var(above)
        })

if variance_windows:
    min_var_point = min(variance_windows, key=lambda x: x['total_var'])
    print(f"\nOptimal split point (minimum variance): {min_var_point['threshold']:.1f}%")

print("\n" + "="*70)
print("TEST 4: ALPHABET ANALYSIS - Clustering states")
print("="*70)

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

features = np.column_stack([
    (theta_centroids - np.mean(theta_centroids)) / np.std(theta_centroids),
    (alpha_centroids - np.mean(alpha_centroids)) / np.std(alpha_centroids),
    (ratios - np.mean(ratios)) / np.std(ratios),
    (convergences - np.mean(convergences)) / np.std(convergences)
])

linkage_matrix = linkage(features, method='ward')

silhouette_scores = []
for n_clusters in range(2, 8):
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    from sklearn.metrics import silhouette_score
    score = silhouette_score(features, clusters)
    silhouette_scores.append((n_clusters, score))
    print(f"  K={n_clusters}: silhouette = {score:.3f}")

best_k = max(silhouette_scores, key=lambda x: x[1])[0]
print(f"\n🏆 Optimal number of states (alphabet size): K = {best_k}")

clusters = fcluster(linkage_matrix, best_k, criterion='maxclust')

print(f"\nCluster characteristics:")
for c in range(1, best_k + 1):
    mask = clusters == c
    print(f"\n  Cluster {c} (N={sum(mask)}):")
    print(f"    Mean θ: {np.mean(theta_centroids[mask]):.3f} Hz")
    print(f"    Mean α: {np.mean(alpha_centroids[mask]):.3f} Hz")
    print(f"    Mean ratio: {np.mean(ratios[mask]):.3f}")
    print(f"    Mean convergence: {np.mean(convergences[mask]):.1f}%")

print("\n" + "="*70)
print("TEST 5: GOLDEN SPIRAL - φ-based frequency progression")
print("="*70)

mean_delta = np.mean(delta_c)
predicted_theta = mean_delta * PHI
predicted_alpha = predicted_theta * PHI
predicted_beta = predicted_alpha * PHI

print(f"\nStarting from δ = {mean_delta:.3f} Hz:")
print(f"  Predicted θ = δ×φ = {predicted_theta:.3f} Hz (Actual: {np.mean(theta_c):.3f})")
print(f"  Predicted α = θ×φ = {predicted_alpha:.3f} Hz (Actual: {np.mean(alpha_c):.3f})")
print(f"  Predicted β = α×φ = {predicted_beta:.3f} Hz (Actual: {np.mean(beta_c):.3f})")

errors = [
    abs(predicted_theta - np.mean(theta_c)),
    abs(predicted_alpha - np.mean(alpha_c)),
    abs(predicted_beta - np.mean(beta_c))
]
print(f"\nPrediction errors: θ={errors[0]:.2f}, α={errors[1]:.2f}, β={errors[2]:.2f}")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

ax1 = axes[0, 0]
ax1.hist(theta_centroids, bins=15, density=True, alpha=0.5, color='green', edgecolor='black', label='Theta')
ax1.hist(alpha_centroids, bins=15, density=True, alpha=0.5, color='blue', edgecolor='black', label='Alpha')
ax1.plot(x_theta, y_theta, 'g-', linewidth=2)
ax1.plot(x_alpha, y_alpha, 'b-', linewidth=2)
ax1.set_xlabel('Frequency (Hz)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('A. Quantum Levels: θ and α Distributions', fontsize=14)
ax1.legend()

ax2 = axes[0, 1]
ax2.hist(convergences, bins=20, density=True, alpha=0.7, color='purple', edgecolor='black')
ax2.plot(x_conv, y_conv, 'r-', linewidth=2)
for p in peaks_conv:
    ax2.axvline(x_conv[p], color='gold', linestyle='--', linewidth=2)
ax2.set_xlabel('8 Hz Convergence (%)', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title('B. Phase Transition: Convergence Distribution', fontsize=14)

ax3 = axes[0, 2]
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink'][:best_k]
for c in range(1, best_k + 1):
    mask = clusters == c
    ax3.scatter(ratios[mask], convergences[mask], c=colors[c-1], s=80, alpha=0.7, 
               label=f'State {c}', edgecolor='black')
ax3.axvline(PHI, color='gold', linestyle='--', linewidth=2, label='φ')
ax3.set_xlabel('θ/α Ratio', fontsize=12)
ax3.set_ylabel('Convergence (%)', fontsize=12)
ax3.set_title(f'C. Alphabet: {best_k} Distinct States', fontsize=14)
ax3.legend(fontsize=9)

ax4 = axes[1, 0]
bands = ['δ', 'θ', 'α', 'β']
actual = [np.mean(delta_c), np.mean(theta_c), np.mean(alpha_c), np.mean(beta_c)]
predicted = [mean_delta, predicted_theta, predicted_alpha, predicted_beta]
x = np.arange(len(bands))
width = 0.35
ax4.bar(x - width/2, actual, width, label='Actual', color='steelblue', edgecolor='black')
ax4.bar(x + width/2, predicted, width, label='φ-predicted', color='gold', edgecolor='black')
ax4.set_xticks(x)
ax4.set_xticklabels(bands)
ax4.set_ylabel('Centroid (Hz)', fontsize=12)
ax4.set_title('D. Golden Spiral: φ-based Prediction', fontsize=14)
ax4.legend()

ax5 = axes[1, 1]
pcis = np.array([d['pci'] for d in all_data])
scatter = ax5.scatter(ratios, convergences, c=pcis, cmap='RdYlGn', s=80, alpha=0.7, edgecolor='black')
plt.colorbar(scatter, ax=ax5, label='PCI')
ax5.axvline(PHI, color='gold', linestyle='--', linewidth=2, label='φ')
ax5.set_xlabel('θ/α Ratio', fontsize=12)
ax5.set_ylabel('Convergence (%)', fontsize=12)
ax5.set_title('E. 3D Structure: Ratio × Convergence × PCI', fontsize=14)
ax5.legend()

ax6 = axes[1, 2]
ax6.axis('off')

summary = f"""
QUANTUM STRUCTURE ANALYSIS SUMMARY
================================================

N = {len(all_data)} subjects

QUANTUM LEVELS:
  Theta peaks: {len(peaks_theta)}
  Alpha peaks: {len(peaks_alpha)}

PHASE TRANSITION:
  Convergence peaks: {len(peaks_conv)}
  {"BIMODAL (2 phases)" if len(peaks_conv) >= 2 else "UNIMODAL"}

ALPHABET SIZE:
  Optimal K = {best_k} distinct brain states

GOLDEN SPIRAL:
  δ={mean_delta:.2f} → θ={predicted_theta:.2f} → α={predicted_alpha:.2f} → β={predicted_beta:.2f}
  (Actual: θ={np.mean(theta_c):.2f}, α={np.mean(alpha_c):.2f}, β={np.mean(beta_c):.2f})

CONSECUTIVE BAND RATIOS:
  θ/δ = {np.mean(theta_c)/np.mean(delta_c):.3f}
  α/θ = {np.mean(alpha_c)/np.mean(theta_c):.3f} (closest to φ!)
  β/α = {np.mean(beta_c)/np.mean(alpha_c):.3f}

φ = {PHI:.6f}
"""

ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('quantum_structure_analysis.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: quantum_structure_analysis.png")
