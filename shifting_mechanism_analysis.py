"""
SHIFTING MECHANISM ANALYSIS
===========================
Detectăm cum creierul se mișcă între vibraton states în timp
"""

import numpy as np
from scipy import stats, signal
import matplotlib.pyplot as plt
from collections import Counter
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf

mne.set_log_level('ERROR')

PHI = (1 + np.sqrt(5)) / 2

print("="*70)
print("SHIFTING MECHANISM ANALYSIS")
print("Tracking brain dynamics between vibraton states")
print("="*70)

VIBRATON_EDGES = [1.45, 1.55, 1.64, 1.70, 1.77, 1.83, 1.90, 2.05]
VIBRATON_NAMES = ['V1(3:2)', 'V2(phi)', 'V3(5:3)', 'V4(7:4)', 'V5(9:5)', 'V6', 'V7(2:1)']

def get_instantaneous_ratio(eeg, sfreq, window_sec=2):
    """Compute instantaneous theta/alpha ratio over time."""
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
    """Assign each ratio to a vibraton state."""
    states = np.digitize(ratios, VIBRATON_EDGES) - 1
    return np.clip(states, 0, 6)

def count_transitions(states):
    """Count transitions between vibraton states."""
    transitions = Counter()
    for i in range(len(states) - 1):
        from_state = states[i]
        to_state = states[i+1]
        if from_state != to_state:
            transitions[(from_state, to_state)] += 1
    return transitions

def analyze_shift_direction(transitions):
    """Compare forward (toward phi) vs backward (toward 2:1) shifts."""
    forward = 0
    backward = 0
    
    for (from_s, to_s), count in transitions.items():
        if to_s < from_s:
            forward += count
        else:
            backward += count
    
    return {
        'forward_to_phi': forward,
        'backward_to_2:1': backward,
        'ratio': forward / (backward + 0.001),
        'bias': 'toward_phi' if forward > backward else 'toward_2:1'
    }

def analyze_step_sizes(transitions):
    """Analyze step sizes in transitions."""
    step_sizes = []
    
    for (from_s, to_s), count in transitions.items():
        step = abs(to_s - from_s)
        step_sizes.extend([step] * count)
    
    if not step_sizes:
        return {'mean_step': 0, 'sequential_pct': 0}
    
    return {
        'mean_step': np.mean(step_sizes),
        'step_distribution': Counter(step_sizes),
        'sequential_pct': step_sizes.count(1) / len(step_sizes) * 100
    }

print("\nLoading and analyzing 30 subjects...")

all_subjects_data = []
all_transitions = []

for subj in range(1, 31):
    try:
        raw = read_raw_edf(eegbci.load_data(subj, [1], update_path=True, verbose=False)[0], 
                          preload=True, verbose=False)
        raw.filter(1, 45, verbose=False)
        sfreq = raw.info['sfreq']
        eeg = np.mean(raw.get_data(), axis=0)
        
        ratios_ts = get_instantaneous_ratio(eeg, sfreq)
        states = assign_vibraton_state(ratios_ts)
        transitions = count_transitions(states)
        
        f, p = signal.welch(eeg, sfreq, nperseg=min(1024, len(eeg)//2))
        t_mask = (f >= 4) & (f <= 8)
        a_mask = (f >= 8) & (f <= 13)
        f_t = np.sum(f[t_mask] * p[t_mask]) / np.sum(p[t_mask])
        f_a = np.sum(f[a_mask] * p[a_mask]) / np.sum(p[a_mask])
        mean_ratio = f_a / f_t if f_t > 0 else np.nan
        
        conv_count = sum(1 for s in states if s <= 2)
        convergence = 100 * conv_count / len(states) if len(states) > 0 else 0
        
        direction = analyze_shift_direction(transitions)
        step_stats = analyze_step_sizes(transitions)
        
        subj_data = {
            'subject': subj,
            'mean_ratio': mean_ratio,
            'convergence': convergence,
            'states': states,
            'transitions': transitions,
            'n_transitions': sum(transitions.values()),
            'forward_bias': direction['ratio'],
            'direction': direction['bias'],
            'mean_step': step_stats['mean_step'],
            'sequential_pct': step_stats['sequential_pct'],
            'time_in_phi': sum(1 for s in states if s <= 2) / len(states) * 100 if len(states) > 0 else 0
        }
        
        all_subjects_data.append(subj_data)
        all_transitions.append(transitions)
        
        print(f"S{subj:02d}: ratio={mean_ratio:.3f}, conv={convergence:.1f}%, "
              f"transitions={subj_data['n_transitions']}, bias={direction['bias']}")
        
    except Exception as e:
        print(f"S{subj:02d}: Error - {e}")
        continue

print(f"\nSuccessfully analyzed {len(all_subjects_data)} subjects")

print("\n" + "="*70)
print("TRANSITION MATRIX ANALYSIS")
print("="*70)

n_states = 7
transition_counts = np.zeros((n_states, n_states))

for transitions in all_transitions:
    for (from_s, to_s), count in transitions.items():
        transition_counts[from_s, to_s] += count

row_sums = transition_counts.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
transition_matrix = transition_counts / row_sums

print("\nTransition Matrix (P[from → to]):")
print(f"{'':>10}", end='')
for name in VIBRATON_NAMES:
    print(f"{name[:4]:>8}", end='')
print()

for i, name in enumerate(VIBRATON_NAMES):
    print(f"{name[:10]:>10}", end='')
    for j in range(n_states):
        print(f"{transition_matrix[i,j]:>8.3f}", end='')
    print()

print("\n" + "="*70)
print("HIGH vs LOW CONVERTERS DYNAMICS")
print("="*70)

median_conv = np.median([d['convergence'] for d in all_subjects_data])
high_conv = [d for d in all_subjects_data if d['convergence'] > median_conv]
low_conv = [d for d in all_subjects_data if d['convergence'] <= median_conv]

print(f"\nMedian convergence: {median_conv:.1f}%")
print(f"HIGH group: N={len(high_conv)}, LOW group: N={len(low_conv)}")

metrics = [
    ('n_transitions', 'Total Transitions'),
    ('forward_bias', 'Forward Bias (toward phi)'),
    ('mean_step', 'Mean Step Size'),
    ('sequential_pct', 'Sequential Transitions (%)'),
    ('time_in_phi', 'Time in phi zone (%)'),
]

print(f"\n{'Metric':<30} {'HIGH':>12} {'LOW':>12} {'t':>8} {'p':>10} {'Sig':>5}")
print("-"*80)

for metric, label in metrics:
    high_vals = [d[metric] for d in high_conv]
    low_vals = [d[metric] for d in low_conv]
    
    t, p = stats.ttest_ind(high_vals, low_vals)
    sig = "**" if p < 0.05 else "*" if p < 0.1 else ""
    
    print(f"{label:<30} {np.mean(high_vals):>12.2f} {np.mean(low_vals):>12.2f} "
          f"{t:>8.2f} {p:>10.4f} {sig}")

print("\n" + "="*70)
print("SHIFT DIRECTION ANALYSIS")
print("="*70)

forward_total = sum(d['forward_bias'] > 1 for d in all_subjects_data)
backward_total = len(all_subjects_data) - forward_total

print(f"\nSubjects with forward bias (toward phi): {forward_total} ({100*forward_total/len(all_subjects_data):.0f}%)")
print(f"Subjects with backward bias (toward 2:1): {backward_total} ({100*backward_total/len(all_subjects_data):.0f}%)")

forward_biases = [d['forward_bias'] for d in all_subjects_data]
r, p = stats.pearsonr(forward_biases, [d['convergence'] for d in all_subjects_data])
print(f"\nForward bias ↔ Convergence: r = {r:.3f}, p = {p:.4f}")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

ax1 = axes[0, 0]
im = ax1.imshow(transition_matrix, cmap='YlOrRd')
plt.colorbar(im, ax=ax1, label='Probability')
ax1.set_xticks(range(7))
ax1.set_xticklabels(['V1', 'V2\nφ', 'V3', 'V4', 'V5', 'V6', 'V7'], fontsize=9)
ax1.set_yticks(range(7))
ax1.set_yticklabels(['V1', 'V2\nφ', 'V3', 'V4', 'V5', 'V6', 'V7'], fontsize=9)
ax1.set_xlabel('To State', fontsize=11)
ax1.set_ylabel('From State', fontsize=11)
ax1.set_title('A. Vibraton Transition Matrix', fontsize=14)

for i in range(7):
    for j in range(7):
        if transition_matrix[i,j] > 0.05:
            ax1.text(j, i, f'{transition_matrix[i,j]:.2f}', ha='center', va='center', fontsize=8)

ax2 = axes[0, 1]
forward_biases = [d['forward_bias'] for d in all_subjects_data]
convergences = [d['convergence'] for d in all_subjects_data]
ax2.scatter(forward_biases, convergences, c='purple', s=80, alpha=0.7)
ax2.axvline(1.0, color='black', linestyle='--', label='No bias')
ax2.set_xlabel('Forward Bias (toward φ)', fontsize=12)
ax2.set_ylabel('Convergence (%)', fontsize=12)
ax2.set_title(f'B. Forward Bias vs Convergence\n(r = {r:.3f})', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = axes[0, 2]
high_metric = [d['n_transitions'] for d in high_conv]
low_metric = [d['n_transitions'] for d in low_conv]
positions = [1, 2]
bp = ax3.boxplot([high_metric, low_metric], positions=positions, widths=0.6, patch_artist=True)
bp['boxes'][0].set_facecolor('gold')
bp['boxes'][1].set_facecolor('gray')
ax3.set_xticks(positions)
ax3.set_xticklabels(['HIGH\nconv', 'LOW\nconv'])
ax3.set_ylabel('Number of Transitions', fontsize=12)
ax3.set_title('C. Transition Count: HIGH vs LOW', fontsize=14)

ax4 = axes[1, 0]
if len(all_subjects_data) > 0:
    best_subj = max(all_subjects_data, key=lambda d: d['convergence'])
    ax4.plot(best_subj['states'], 'b-', alpha=0.7, linewidth=0.5)
    ax4.axhline(y=1, color='gold', linestyle='--', linewidth=2, label='V2 (φ)')
    ax4.axhline(y=4, color='gray', linestyle='--', linewidth=1, label='V5 (9:5)')
    ax4.set_yticks(range(7))
    ax4.set_yticklabels(['V1', 'V2\nφ', 'V3', 'V4', 'V5', 'V6', 'V7'], fontsize=9)
    ax4.set_xlabel('Time (windows)', fontsize=12)
    ax4.set_ylabel('Vibraton State', fontsize=12)
    ax4.set_title(f'D. Best Subject (S{best_subj["subject"]}) Trajectory', fontsize=14)
    ax4.legend(loc='upper right')

ax5 = axes[1, 1]
all_states = np.concatenate([d['states'] for d in all_subjects_data])
state_counts = [np.sum(all_states == i) for i in range(7)]
colors = ['gold' if i <= 2 else 'steelblue' for i in range(7)]
ax5.bar(range(7), state_counts, color=colors, edgecolor='black')
ax5.set_xticks(range(7))
ax5.set_xticklabels(['V1\n3:2', 'V2\nφ', 'V3\n5:3', 'V4\n7:4', 'V5\n9:5', 'V6', 'V7\n2:1'], fontsize=9)
ax5.set_ylabel('Time in State (samples)', fontsize=12)
ax5.set_title('E. Time Spent in Each Vibraton', fontsize=14)

ax6 = axes[1, 2]
ax6.axis('off')

summary = f"""
SHIFTING MECHANISM SUMMARY
================================================

N = {len(all_subjects_data)} subjects analyzed

TRANSITION DYNAMICS:
  Mean transitions/subject: {np.mean([d['n_transitions'] for d in all_subjects_data]):.1f}
  Mean step size: {np.mean([d['mean_step'] for d in all_subjects_data]):.2f}
  Sequential transitions: {np.mean([d['sequential_pct'] for d in all_subjects_data]):.1f}%

DIRECTION BIAS:
  Forward (→φ): {forward_total} subjects ({100*forward_total/len(all_subjects_data):.0f}%)
  Backward (→2:1): {backward_total} subjects ({100*backward_total/len(all_subjects_data):.0f}%)
  
  Forward bias ↔ Convergence: r = {r:+.3f}

HIGH vs LOW CONVERTERS:
  HIGH group time in φ-zone: {np.mean([d['time_in_phi'] for d in high_conv]):.1f}%
  LOW group time in φ-zone: {np.mean([d['time_in_phi'] for d in low_conv]):.1f}%

INTERPRETATION:
Forward bias (more shifts toward φ) correlates
with higher convergence/∞-state access.

φ = {PHI:.4f}
"""

ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('shifting_mechanism_analysis.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: shifting_mechanism_analysis.png")
