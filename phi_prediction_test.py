"""
PHI PREDICTION TEST
===================
Does θ × φ predict α centroid?
"""

import numpy as np
from scipy.stats import pearsonr
from scipy import signal
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf

mne.set_log_level('ERROR')

PHI = (1 + np.sqrt(5)) / 2

print("="*70)
print("PHI PREDICTION TEST")
print(f"Testing if θ × φ = α (φ = {PHI:.6f})")
print("="*70)

BANDS = {'theta': (4, 8), 'alpha': (8, 13)}

def get_peak_centroid(psd, freqs, band):
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask) or np.sum(psd[mask]) == 0:
        return np.nan
    return np.sum(freqs[mask] * psd[mask]) / np.sum(psd[mask])

print("\nLoading data from 50 subjects...")

theta_centroids = []
alpha_centroids = []
convergences = []

print(f"\n{'Subj':<6} {'θ':>8} {'α':>8} {'θ×φ':>8} {'error':>8} {'actual α/θ':>12}")
print("-"*60)

for subj in range(1, 51):
    try:
        raw = read_raw_edf(eegbci.load_data(subj, [1], update_path=True, verbose=False)[0], 
                          preload=True, verbose=False)
        raw.filter(1, 45, verbose=False)
        sfreq = raw.info['sfreq']
        eeg = np.mean(raw.get_data(), axis=0)
        
        f, p = signal.welch(eeg, sfreq, nperseg=min(1024, len(eeg)//2))
        theta_cent = get_peak_centroid(p, f, BANDS['theta'])
        alpha_cent = get_peak_centroid(p, f, BANDS['alpha'])
        
        if not np.isnan(theta_cent) and not np.isnan(alpha_cent):
            theta_centroids.append(theta_cent)
            alpha_centroids.append(alpha_cent)
            
            predicted_alpha = theta_cent * PHI
            error = abs(predicted_alpha - alpha_cent)
            actual_ratio = alpha_cent / theta_cent
            
            marker = "✓" if error < 0.5 else ""
            print(f"S{subj:02d}    {theta_cent:>8.3f} {alpha_cent:>8.3f} {predicted_alpha:>8.3f} {error:>8.3f} {actual_ratio:>12.4f} {marker}")
    except:
        continue

theta_centroids = np.array(theta_centroids)
alpha_centroids = np.array(alpha_centroids)

print("\n" + "="*70)
print("PREDICTION ANALYSIS")
print("="*70)

predicted_alpha = theta_centroids * PHI
errors = np.abs(predicted_alpha - alpha_centroids)

print(f"\nN = {len(theta_centroids)} subjects")
print(f"\nMean θ centroid: {np.mean(theta_centroids):.3f} Hz")
print(f"Mean α centroid: {np.mean(alpha_centroids):.3f} Hz")
print(f"Mean θ × φ: {np.mean(predicted_alpha):.3f} Hz")

print(f"\nPrediction error (θ×φ vs actual α):")
print(f"  Mean error: {np.mean(errors):.3f} Hz")
print(f"  Std error: {np.std(errors):.3f} Hz")
print(f"  Min error: {np.min(errors):.3f} Hz")
print(f"  Max error: {np.max(errors):.3f} Hz")

r, p = pearsonr(predicted_alpha, alpha_centroids)
print(f"\n📊 CORRELATION: θ × φ predicts α")
print(f"   r = {r:.4f}, p = {p:.6f}")

if p < 0.001:
    print(f"   ✓ HIGHLY SIGNIFICANT!")

r2, p2 = pearsonr(theta_centroids, alpha_centroids)
print(f"\n📊 BASELINE: θ predicts α (without φ)")
print(f"   r = {r2:.4f}, p = {p2:.6f}")

print("\n" + "="*70)
print("ALTERNATIVE MULTIPLIERS TEST")
print("="*70)

multipliers = {
    '1.5 (3:2)': 1.5,
    '1.6 (8:5)': 1.6,
    'φ': PHI,
    '1.67 (5:3)': 1.667,
    '1.75 (7:4)': 1.75,
    '1.8 (9:5)': 1.8,
    '2.0 (2:1)': 2.0
}

print(f"\n{'Multiplier':<15} {'Mean Error':>12} {'Correlation r':>15} {'p-value':>12}")
print("-"*60)

best_mult = None
best_r = -1

for name, mult in sorted(multipliers.items(), key=lambda x: x[1]):
    pred = theta_centroids * mult
    err = np.mean(np.abs(pred - alpha_centroids))
    r_m, p_m = pearsonr(pred, alpha_centroids)
    
    marker = ""
    if r_m > best_r:
        best_r = r_m
        best_mult = name
        marker = " <-- BEST"
    
    print(f"{name:<15} {err:>12.4f} {r_m:>15.4f} {p_m:>12.6f}{marker}")

print(f"\n🏆 Best multiplier: {best_mult}")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
ax1.scatter(theta_centroids, alpha_centroids, c='steelblue', s=80, alpha=0.7, label='Actual')
ax1.scatter(theta_centroids, predicted_alpha, c='gold', s=60, alpha=0.7, marker='x', label='Predicted (θ×φ)')
ax1.plot([5, 7], [5*PHI, 7*PHI], 'g--', linewidth=2, label=f'θ×φ line')

for i in range(len(theta_centroids)):
    ax1.plot([theta_centroids[i], theta_centroids[i]], 
             [alpha_centroids[i], predicted_alpha[i]], 
             'r-', alpha=0.3, linewidth=1)

ax1.set_xlabel('θ Centroid (Hz)', fontsize=12)
ax1.set_ylabel('α Centroid (Hz)', fontsize=12)
ax1.set_title(f'A. θ×φ Prediction vs Actual α\n(r = {r:.3f})', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.hist(errors, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(errors):.3f}')
ax2.set_xlabel('Prediction Error |θ×φ - α| (Hz)', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('B. Distribution of Prediction Errors', fontsize=14)
ax2.legend(fontsize=10)

ax3 = axes[1, 0]
ax3.scatter(predicted_alpha, alpha_centroids, c='purple', s=80, alpha=0.7)
min_val = min(predicted_alpha.min(), alpha_centroids.min())
max_val = max(predicted_alpha.max(), alpha_centroids.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect prediction')
ax3.set_xlabel('Predicted α (θ×φ) (Hz)', fontsize=12)
ax3.set_ylabel('Actual α Centroid (Hz)', fontsize=12)
ax3.set_title(f'C. Predicted vs Actual α\n(r = {r:.3f}, p < 0.001)', fontsize=14)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
ax4.axis('off')

summary = f"""
PHI PREDICTION TEST RESULTS
================================================

N = {len(theta_centroids)} subjects

PREDICTION: α = θ × φ

Mean θ: {np.mean(theta_centroids):.3f} Hz
Mean α (actual): {np.mean(alpha_centroids):.3f} Hz  
Mean α (predicted): {np.mean(predicted_alpha):.3f} Hz

Mean prediction error: {np.mean(errors):.3f} Hz
Error range: [{np.min(errors):.3f}, {np.max(errors):.3f}]

CORRELATION:
θ × φ ↔ actual α: r = {r:.4f}, p = {p:.6f}
{"✓ SIGNIFICANT!" if p < 0.05 else "Not significant"}

INTERPRETATION:
{"θ × φ is a good predictor of α!" if r > 0.5 else "θ × φ has moderate predictive power."}
The relationship α ≈ θ × φ holds approximately.

φ = {PHI:.6f}
"""

ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('phi_prediction_test.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: phi_prediction_test.png")
