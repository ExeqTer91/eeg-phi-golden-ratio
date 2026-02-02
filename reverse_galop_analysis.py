"""
REVERSE GALOP & ESCAPE ANALYSIS
================================
Câte salturi mari merg spre φ vs spre 2:1?
Cine scapă din lock-zone?
"""

import numpy as np
from collections import Counter
from scipy import stats, signal
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf

mne.set_log_level('ERROR')

PHI = (1 + np.sqrt(5)) / 2

print("="*70)
print("REVERSE GALOP & ESCAPE ANALYSIS")
print("Tracking directional big jumps and lock-zone escapes")
print("="*70)

VIBRATON_EDGES = [1.45, 1.55, 1.64, 1.70, 1.77, 1.83, 1.90, 2.05]
VIBRATON_NAMES = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7']

def get_instantaneous_ratio(eeg, sfreq, window_sec=2):
    window = int(window_sec * sfreq)
    step = window // 4
    ratios = []
    
    for start in range(0, len(eeg) - window, step):
        seg = eeg[start:start+window]
        f, p = signal.welch(seg, sfreq, nperseg=min(512, len(seg)))
        
        t_mask = (f >= 4) & (f <= 8)
        a_mask = (f >= 8) & (f <= 13)
        
        if np.any(t_mask) and np.any(a_mask) and np.sum(p[t_mask]) > 0:
            f_t = np.sum(f[t_mask] * p[t_mask]) / np.sum(p[t_mask])
            f_a = np.sum(f[a_mask] * p[a_mask]) / np.sum(p[a_mask])
            if f_t > 0:
                ratios.append(f_a / f_t)
    
    return np.array(ratios)

def assign_vibraton_state(ratios):
    states = np.digitize(ratios, VIBRATON_EDGES) - 1
    return np.clip(states, 0, 6)

print("\nLoading and analyzing 30 subjects...")

all_transitions = Counter()
all_subjects_data = []

for subj in range(1, 31):
    try:
        raw = read_raw_edf(eegbci.load_data(subj, [1], update_path=True, verbose=False)[0], 
                          preload=True, verbose=False)
        raw.filter(1, 45, verbose=False)
        sfreq = raw.info['sfreq']
        eeg = np.mean(raw.get_data(), axis=0)
        
        ratios_ts = get_instantaneous_ratio(eeg, sfreq)
        states = assign_vibraton_state(ratios_ts)
        
        subj_transitions = Counter()
        for i in range(len(states) - 1):
            if states[i] != states[i+1]:
                subj_transitions[(states[i], states[i+1])] += 1
                all_transitions[(states[i], states[i+1])] += 1
        
        conv = sum(1 for s in states if s <= 2) / len(states) * 100 if len(states) > 0 else 0
        
        forward_galop = 0
        backward_galop = 0
        
        for (from_s, to_s), count in subj_transitions.items():
            step = abs(to_s - from_s)
            if step >= 3:
                if to_s < from_s:
                    forward_galop += count
                else:
                    backward_galop += count
        
        escape_count = 0
        for (from_s, to_s), count in subj_transitions.items():
            if from_s >= 5 and to_s <= 2:
                escape_count += count
        
        all_subjects_data.append({
            'subject': subj,
            'convergence': conv,
            'forward_galop': forward_galop,
            'backward_galop': backward_galop,
            'escape_count': escape_count,
            'galop_ratio': forward_galop / (backward_galop + 0.001)
        })
        
    except Exception as e:
        print(f"S{subj:02d}: Error - {e}")
        continue

print(f"Analyzed {len(all_subjects_data)} subjects")

print("\n" + "="*70)
print("REVERSE GALOP ANALYSIS (Big jumps, step >= 3)")
print("="*70)

total_forward_galop = 0
total_backward_galop = 0

for (from_s, to_s), count in all_transitions.items():
    step = abs(to_s - from_s)
    if step >= 3:
        if to_s < from_s:
            total_forward_galop += count
        else:
            total_backward_galop += count

print(f"\nTotal big jumps (step >= 3):")
print(f"  Forward galop (→φ): {total_forward_galop}")
print(f"  Backward galop (→2:1): {total_backward_galop}")
print(f"  Ratio (forward/backward): {total_forward_galop/(total_backward_galop+0.001):.2f}")

if total_forward_galop > total_backward_galop:
    print(f"\n✓ MORE jumps TOWARD φ than toward 2:1!")
else:
    print(f"\n⚠ More jumps toward 2:1 than toward φ")

print("\n" + "="*70)
print("ESCAPE ANALYSIS (Lock zone → φ zone)")
print("="*70)

escape_transitions = {}
for from_s in [5, 6]:
    for to_s in [0, 1, 2]:
        key = f"V{from_s+1}→V{to_s+1}"
        escape_transitions[key] = all_transitions.get((from_s, to_s), 0)

print(f"\nESCAPE TRANSITIONS (Lock zone V6-V7 → φ zone V1-V3):")
total_escapes = 0
for key, count in sorted(escape_transitions.items()):
    print(f"  {key}: {count}")
    total_escapes += count

print(f"\n  Total escapes: {total_escapes}")

trap_transitions = {}
for from_s in [0, 1, 2]:
    for to_s in [5, 6]:
        key = f"V{from_s+1}→V{to_s+1}"
        trap_transitions[key] = all_transitions.get((from_s, to_s), 0)

print(f"\nTRAP TRANSITIONS (φ zone → Lock zone):")
total_traps = 0
for key, count in sorted(trap_transitions.items()):
    print(f"  {key}: {count}")
    total_traps += count

print(f"\n  Total traps: {total_traps}")
print(f"\n  ESCAPE/TRAP ratio: {total_escapes/(total_traps+0.001):.2f}")

print("\n" + "="*70)
print("HIGH vs LOW CONVERTERS: Escape Ability")
print("="*70)

median_conv = np.median([d['convergence'] for d in all_subjects_data])
high_conv = [d for d in all_subjects_data if d['convergence'] > median_conv]
low_conv = [d for d in all_subjects_data if d['convergence'] <= median_conv]

print(f"\nMedian convergence: {median_conv:.1f}%")

metrics = [
    ('forward_galop', 'Forward galop (→φ)'),
    ('backward_galop', 'Backward galop (→2:1)'),
    ('galop_ratio', 'Galop ratio (fwd/bwd)'),
    ('escape_count', 'Escape count'),
]

print(f"\n{'Metric':<30} {'HIGH':>10} {'LOW':>10} {'t':>8} {'p':>10} {'Sig':>5}")
print("-"*75)

for metric, label in metrics:
    high_vals = [d[metric] for d in high_conv]
    low_vals = [d[metric] for d in low_conv]
    
    t, p = stats.ttest_ind(high_vals, low_vals)
    sig = "**" if p < 0.05 else "*" if p < 0.1 else ""
    
    print(f"{label:<30} {np.mean(high_vals):>10.2f} {np.mean(low_vals):>10.2f} "
          f"{t:>8.2f} {p:>10.4f} {sig}")

print("\n" + "="*70)
print("CORRELATION ANALYSIS")
print("="*70)

convergences = [d['convergence'] for d in all_subjects_data]
forward_galops = [d['forward_galop'] for d in all_subjects_data]
escape_counts = [d['escape_count'] for d in all_subjects_data]
galop_ratios = [d['galop_ratio'] for d in all_subjects_data]

r1, p1 = stats.pearsonr(forward_galops, convergences)
r2, p2 = stats.pearsonr(escape_counts, convergences)
r3, p3 = stats.pearsonr(galop_ratios, convergences)

print(f"\nForward galop ↔ Convergence: r = {r1:+.3f}, p = {p1:.4f}")
print(f"Escape count ↔ Convergence: r = {r2:+.3f}, p = {p2:.4f}")
print(f"Galop ratio ↔ Convergence: r = {r3:+.3f}, p = {p3:.4f}")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

ax1 = axes[0, 0]
labels = ['Forward\n(→φ)', 'Backward\n(→2:1)']
counts = [total_forward_galop, total_backward_galop]
colors = ['gold', 'red']
ax1.bar(labels, counts, color=colors, edgecolor='black')
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('A. Big Jumps (step >= 3) Direction', fontsize=14)
for i, (c, v) in enumerate(zip(labels, counts)):
    ax1.text(i, v + 5, str(v), ha='center', fontsize=12, fontweight='bold')

ax2 = axes[0, 1]
escape_keys = list(escape_transitions.keys())
escape_vals = list(escape_transitions.values())
ax2.bar(escape_keys, escape_vals, color='green', edgecolor='black')
ax2.set_xlabel('Transition', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('B. Escape Transitions (Lock → φ)', fontsize=14)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

ax3 = axes[0, 2]
ax3.bar(['Escapes', 'Traps'], [total_escapes, total_traps], color=['green', 'red'], edgecolor='black')
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title(f'C. Escape vs Trap\n(ratio = {total_escapes/(total_traps+0.001):.2f})', fontsize=14)

ax4 = axes[1, 0]
ax4.scatter(forward_galops, convergences, c='gold', s=80, alpha=0.7, label='Forward')
ax4.set_xlabel('Forward Galop Count', fontsize=12)
ax4.set_ylabel('Convergence (%)', fontsize=12)
ax4.set_title(f'D. Forward Galop vs Convergence\n(r = {r1:.3f})', fontsize=14)

ax5 = axes[1, 1]
ax5.scatter(escape_counts, convergences, c='green', s=80, alpha=0.7)
ax5.set_xlabel('Escape Count', fontsize=12)
ax5.set_ylabel('Convergence (%)', fontsize=12)
ax5.set_title(f'E. Escape Count vs Convergence\n(r = {r2:.3f})', fontsize=14)

ax6 = axes[1, 2]
ax6.axis('off')

summary = f"""
REVERSE GALOP & ESCAPE SUMMARY
================================================

N = {len(all_subjects_data)} subjects

BIG JUMPS (step >= 3):
  Forward (→φ): {total_forward_galop}
  Backward (→2:1): {total_backward_galop}
  Ratio: {total_forward_galop/(total_backward_galop+0.001):.2f}

ESCAPE vs TRAP:
  Escapes (Lock→φ): {total_escapes}
  Traps (φ→Lock): {total_traps}
  Escape/Trap ratio: {total_escapes/(total_traps+0.001):.2f}

TOP ESCAPE ROUTES:
"""
for key, count in sorted(escape_transitions.items(), key=lambda x: -x[1])[:3]:
    summary += f"\n  {key}: {count}"

summary += f"""

CORRELATIONS WITH CONVERGENCE:
  Forward galop: r = {r1:+.3f}
  Escape count: r = {r2:+.3f}
  Galop ratio: r = {r3:+.3f}
"""

ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('reverse_galop_analysis.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: reverse_galop_analysis.png")
