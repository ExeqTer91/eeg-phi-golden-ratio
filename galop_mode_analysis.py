"""
GALOP MODE ANALYSIS
===================
Căutăm pattern-uri de mișcare non-secvențială/ritmică
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
print("GALOP MODE ANALYSIS")
print("Looking for rhythmic, non-sequential movement patterns")
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

def analyze_skips(transitions):
    results = {'step_1': 0, 'step_2': 0, 'step_3': 0, 'step_4+': 0}
    
    for (from_s, to_s), count in transitions.items():
        step = abs(to_s - from_s)
        if step == 1:
            results['step_1'] += count
        elif step == 2:
            results['step_2'] += count
        elif step == 3:
            results['step_3'] += count
        else:
            results['step_4+'] += count
    
    return results

def detect_oscillations(state_sequence):
    oscillations = []
    for i in range(len(state_sequence) - 3):
        s1, s2, s3, s4 = state_sequence[i:i+4]
        if s1 == s3 and s2 == s4 and s1 != s2:
            oscillations.append((s1, s2))
    return Counter(oscillations)

def detect_bursts(state_sequence, window=10):
    transition_counts = []
    for i in range(0, len(state_sequence) - window, window):
        segment = state_sequence[i:i+window]
        transitions_in_window = sum(1 for j in range(len(segment)-1) if segment[j] != segment[j+1])
        transition_counts.append(transitions_in_window)
    
    if not transition_counts:
        return {'burst_windows': 0, 'stable_windows': 0, 'burst_ratio': 0, 'mean_transitions': 0}
    
    bursts = sum(1 for t in transition_counts if t > 5)
    stable = sum(1 for t in transition_counts if t < 2)
    
    return {
        'burst_windows': bursts,
        'stable_windows': stable,
        'burst_ratio': bursts / len(transition_counts),
        'mean_transitions': np.mean(transition_counts)
    }

def detect_galop_pattern(state_sequence):
    galop_patterns = {'AAB': 0, 'ABB': 0, 'ABA': 0, 'ABC': 0}
    
    for i in range(len(state_sequence) - 2):
        s1, s2, s3 = state_sequence[i:i+3]
        
        if s1 == s2 and s2 != s3:
            galop_patterns['AAB'] += 1
        elif s1 != s2 and s2 == s3:
            galop_patterns['ABB'] += 1
        elif s1 != s2 and s2 != s3 and s1 == s3:
            galop_patterns['ABA'] += 1
        elif s1 != s2 and s2 != s3 and s1 != s3:
            galop_patterns['ABC'] += 1
    
    return galop_patterns

def analyze_fibonacci_transitions(transitions):
    fib_states = [0, 1, 2]
    non_fib = [3, 4, 5, 6]
    
    fib_to_fib = 0
    fib_to_nonfib = 0
    nonfib_to_fib = 0
    nonfib_to_nonfib = 0
    
    for (from_s, to_s), count in transitions.items():
        if from_s in fib_states and to_s in fib_states:
            fib_to_fib += count
        elif from_s in fib_states and to_s in non_fib:
            fib_to_nonfib += count
        elif from_s in non_fib and to_s in fib_states:
            nonfib_to_fib += count
        else:
            nonfib_to_nonfib += count
    
    return {
        'fib_to_fib': fib_to_fib,
        'fib_to_nonfib': fib_to_nonfib,
        'nonfib_to_fib': nonfib_to_fib,
        'nonfib_to_nonfib': nonfib_to_nonfib,
        'fib_stability': fib_to_fib / (fib_to_fib + fib_to_nonfib + 0.001),
        'nonfib_stability': nonfib_to_nonfib / (nonfib_to_nonfib + nonfib_to_fib + 0.001)
    }

def analyze_zone_transitions(states):
    def get_zone(state):
        if state in [0, 1, 2]:
            return 'phi'
        elif state in [3, 4]:
            return 'Mid'
        else:
            return 'Lock'
    
    zone_sequence = [get_zone(s) for s in states]
    zone_transitions = Counter()
    
    for i in range(len(zone_sequence) - 1):
        if zone_sequence[i] != zone_sequence[i+1]:
            zone_transitions[(zone_sequence[i], zone_sequence[i+1])] += 1
    
    return zone_transitions

print("\nLoading and analyzing 30 subjects...")

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
        
        transitions = Counter()
        for i in range(len(states) - 1):
            if states[i] != states[i+1]:
                transitions[(states[i], states[i+1])] += 1
        
        skips = analyze_skips(transitions)
        oscillations = detect_oscillations(states)
        bursts = detect_bursts(states)
        galop = detect_galop_pattern(states)
        fib_trans = analyze_fibonacci_transitions(transitions)
        zone_trans = analyze_zone_transitions(states)
        
        conv = sum(1 for s in states if s <= 2) / len(states) * 100 if len(states) > 0 else 0
        
        all_subjects_data.append({
            'subject': subj,
            'states': states,
            'convergence': conv,
            'skips': skips,
            'oscillations': oscillations,
            'bursts': bursts,
            'galop': galop,
            'fib_trans': fib_trans,
            'zone_trans': zone_trans,
            'sequential_pct': 100 * skips['step_1'] / sum(skips.values()) if sum(skips.values()) > 0 else 0
        })
        
    except Exception as e:
        print(f"S{subj:02d}: Error - {e}")
        continue

print(f"Analyzed {len(all_subjects_data)} subjects")

print("\n" + "="*70)
print("TEST 1: SKIP ANALYSIS (Step Sizes)")
print("="*70)

total_skips = {'step_1': 0, 'step_2': 0, 'step_3': 0, 'step_4+': 0}
for d in all_subjects_data:
    for k, v in d['skips'].items():
        total_skips[k] += v

total_trans = sum(total_skips.values())
print(f"\nStep size distribution (N={total_trans} transitions):")
for step, count in total_skips.items():
    print(f"  {step}: {count} ({100*count/total_trans:.1f}%)")

print("\n" + "="*70)
print("TEST 2: GALOP PATTERNS (3-beat rhythms)")
print("="*70)

total_galop = {'AAB': 0, 'ABB': 0, 'ABA': 0, 'ABC': 0}
for d in all_subjects_data:
    for k, v in d['galop'].items():
        total_galop[k] += v

print(f"\nGalop pattern distribution:")
for pattern, count in total_galop.items():
    print(f"  {pattern}: {count}")

print("\n  AAB = Stay-stay-jump (dwelling then move)")
print("  ABB = Jump-stay-stay (move then dwell)")
print("  ABA = Bounce (return to origin)")
print("  ABC = Sequential progression")

print("\n" + "="*70)
print("TEST 3: FIBONACCI ZONE TRANSITIONS")
print("="*70)

total_fib = {'fib_to_fib': 0, 'fib_to_nonfib': 0, 'nonfib_to_fib': 0, 'nonfib_to_nonfib': 0}
for d in all_subjects_data:
    for k in total_fib.keys():
        total_fib[k] += d['fib_trans'][k]

print(f"\nFibonacci zone (V1-V3) transitions:")
print(f"  φ → φ (stable in Fib zone): {total_fib['fib_to_fib']}")
print(f"  φ → non-φ (leaving Fib zone): {total_fib['fib_to_nonfib']}")
print(f"  non-φ → φ (entering Fib zone): {total_fib['nonfib_to_fib']}")
print(f"  non-φ → non-φ (stable outside): {total_fib['nonfib_to_nonfib']}")

fib_stability = total_fib['fib_to_fib'] / (total_fib['fib_to_fib'] + total_fib['fib_to_nonfib'] + 0.001)
nonfib_stability = total_fib['nonfib_to_nonfib'] / (total_fib['nonfib_to_nonfib'] + total_fib['nonfib_to_fib'] + 0.001)
print(f"\n  φ-zone stability: {100*fib_stability:.1f}%")
print(f"  non-φ stability: {100*nonfib_stability:.1f}%")

print("\n" + "="*70)
print("TEST 4: HIGH vs LOW CONVERTERS GALOP")
print("="*70)

median_conv = np.median([d['convergence'] for d in all_subjects_data])
high_conv = [d for d in all_subjects_data if d['convergence'] > median_conv]
low_conv = [d for d in all_subjects_data if d['convergence'] <= median_conv]

print(f"\nMedian convergence: {median_conv:.1f}%")

metrics = [
    ('sequential_pct', 'Sequential transitions (%)'),
    ('bursts.burst_ratio', 'Burst ratio'),
]

for metric_key, label in metrics:
    if '.' in metric_key:
        main, sub = metric_key.split('.')
        high_vals = [d[main][sub] for d in high_conv]
        low_vals = [d[main][sub] for d in low_conv]
    else:
        high_vals = [d[metric_key] for d in high_conv]
        low_vals = [d[metric_key] for d in low_conv]
    
    t, p = stats.ttest_ind(high_vals, low_vals)
    sig = "**" if p < 0.05 else "*" if p < 0.1 else ""
    print(f"\n{label}:")
    print(f"  HIGH: {np.mean(high_vals):.2f}, LOW: {np.mean(low_vals):.2f}, p = {p:.4f} {sig}")

high_galop_total = {'AAB': 0, 'ABB': 0, 'ABA': 0, 'ABC': 0}
low_galop_total = {'AAB': 0, 'ABB': 0, 'ABA': 0, 'ABC': 0}

for d in high_conv:
    for k, v in d['galop'].items():
        high_galop_total[k] += v

for d in low_conv:
    for k, v in d['galop'].items():
        low_galop_total[k] += v

print(f"\nGalop patterns comparison:")
print(f"{'Pattern':<10} {'HIGH':>10} {'LOW':>10} {'HIGH%':>10} {'LOW%':>10}")
print("-"*50)

high_total = sum(high_galop_total.values())
low_total = sum(low_galop_total.values())

for pattern in ['AAB', 'ABB', 'ABA', 'ABC']:
    h = high_galop_total[pattern]
    l = low_galop_total[pattern]
    hp = 100 * h / high_total if high_total > 0 else 0
    lp = 100 * l / low_total if low_total > 0 else 0
    print(f"{pattern:<10} {h:>10} {l:>10} {hp:>9.1f}% {lp:>9.1f}%")

print("\n" + "="*70)
print("TEST 5: OSCILLATION PATTERNS")
print("="*70)

all_oscillations = Counter()
for d in all_subjects_data:
    all_oscillations.update(d['oscillations'])

print("\nMost common oscillation pairs (A↔B↔A↔B):")
for (s1, s2), count in all_oscillations.most_common(10):
    print(f"  V{s1+1} ↔ V{s2+1}: {count} times")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

ax1 = axes[0, 0]
steps = list(total_skips.keys())
counts = list(total_skips.values())
colors = ['green', 'yellow', 'orange', 'red']
ax1.bar(steps, counts, color=colors, edgecolor='black')
ax1.set_xlabel('Step Size', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('A. Step Size Distribution', fontsize=14)

ax2 = axes[0, 1]
patterns = list(total_galop.keys())
g_counts = list(total_galop.values())
ax2.bar(patterns, g_counts, color='steelblue', edgecolor='black')
ax2.set_xlabel('Galop Pattern', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('B. Galop Patterns (3-beat rhythms)', fontsize=14)

ax3 = axes[0, 2]
fib_labels = ['φ→φ', 'φ→non', 'non→φ', 'non→non']
fib_counts = [total_fib['fib_to_fib'], total_fib['fib_to_nonfib'], 
              total_fib['nonfib_to_fib'], total_fib['nonfib_to_nonfib']]
colors = ['gold', 'orange', 'lightblue', 'gray']
ax3.bar(fib_labels, fib_counts, color=colors, edgecolor='black')
ax3.set_xlabel('Transition Type', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title('C. Fibonacci Zone Transitions', fontsize=14)

ax4 = axes[1, 0]
x = np.arange(4)
width = 0.35
h_pcts = [100 * high_galop_total[p] / high_total for p in ['AAB', 'ABB', 'ABA', 'ABC']]
l_pcts = [100 * low_galop_total[p] / low_total for p in ['AAB', 'ABB', 'ABA', 'ABC']]
ax4.bar(x - width/2, h_pcts, width, label='HIGH', color='gold', edgecolor='black')
ax4.bar(x + width/2, l_pcts, width, label='LOW', color='gray', edgecolor='black')
ax4.set_xticks(x)
ax4.set_xticklabels(['AAB', 'ABB', 'ABA', 'ABC'])
ax4.set_ylabel('Percentage', fontsize=12)
ax4.set_title('D. Galop: HIGH vs LOW Converters', fontsize=14)
ax4.legend()

ax5 = axes[1, 1]
convergences = [d['convergence'] for d in all_subjects_data]
sequential_pcts = [d['sequential_pct'] for d in all_subjects_data]
ax5.scatter(sequential_pcts, convergences, c='purple', s=80, alpha=0.7)
r, p = stats.pearsonr(sequential_pcts, convergences)
ax5.set_xlabel('Sequential Transitions (%)', fontsize=12)
ax5.set_ylabel('Convergence (%)', fontsize=12)
ax5.set_title(f'E. Sequential vs Convergence\n(r = {r:.3f})', fontsize=14)

ax6 = axes[1, 2]
ax6.axis('off')

summary = f"""
GALOP MODE ANALYSIS SUMMARY
================================================

N = {len(all_subjects_data)} subjects

STEP SIZES:
  Sequential (step 1): {100*total_skips['step_1']/total_trans:.1f}%
  Skip 1 (step 2): {100*total_skips['step_2']/total_trans:.1f}%
  Skip 2 (step 3): {100*total_skips['step_3']/total_trans:.1f}%
  Big jump (step 4+): {100*total_skips['step_4+']/total_trans:.1f}%

GALOP PATTERNS:
  AAB (dwell-dwell-jump): {total_galop['AAB']}
  ABB (jump-dwell-dwell): {total_galop['ABB']}
  ABA (bounce): {total_galop['ABA']}
  ABC (progressive): {total_galop['ABC']}

FIBONACCI ZONES:
  φ-zone stability: {100*fib_stability:.1f}%
  non-φ stability: {100*nonfib_stability:.1f}%

TOP OSCILLATIONS:
"""
for (s1, s2), count in all_oscillations.most_common(3):
    summary += f"\n  V{s1+1}↔V{s2+1}: {count}x"

ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('galop_mode_analysis.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: galop_mode_analysis.png")
