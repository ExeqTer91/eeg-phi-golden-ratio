"""
SHADOW DETECTION ANALYSIS
=========================
Căutăm "umbra" în dinamica neurală
"""

import numpy as np
from scipy import stats, signal
from collections import Counter
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf

mne.set_log_level('ERROR')

PHI = (1 + np.sqrt(5)) / 2

print("="*70)
print("SHADOW DETECTION ANALYSIS")
print("Looking for hidden patterns and shadow states")
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

def get_power_ratio(eeg, sfreq):
    f, p = signal.welch(eeg, sfreq, nperseg=min(1024, len(eeg)//2))
    t_mask = (f >= 4) & (f <= 8)
    a_mask = (f >= 8) & (f <= 13)
    theta_pow = np.sum(p[t_mask])
    alpha_pow = np.sum(p[a_mask])
    return alpha_pow / (theta_pow + 1e-10)

print("\nLoading and analyzing 30 subjects...")

all_ratios = []
all_convergences = []
all_states = []
all_dwell_times = []
all_transitions = Counter()
all_power_ratios = []

for subj in range(1, 31):
    try:
        raw = read_raw_edf(eegbci.load_data(subj, [1], update_path=True, verbose=False)[0], 
                          preload=True, verbose=False)
        raw.filter(1, 45, verbose=False)
        sfreq = raw.info['sfreq']
        eeg = np.mean(raw.get_data(), axis=0)
        
        f, p = signal.welch(eeg, sfreq, nperseg=min(1024, len(eeg)//2))
        t_mask = (f >= 4) & (f <= 8)
        a_mask = (f >= 8) & (f <= 13)
        f_t = np.sum(f[t_mask] * p[t_mask]) / np.sum(p[t_mask])
        f_a = np.sum(f[a_mask] * p[a_mask]) / np.sum(p[a_mask])
        ratio = f_a / f_t if f_t > 0 else np.nan
        
        ratios_ts = get_instantaneous_ratio(eeg, sfreq)
        states = assign_vibraton_state(ratios_ts)
        
        conv = sum(1 for s in states if s <= 2) / len(states) * 100 if len(states) > 0 else 0
        
        for i in range(len(states) - 1):
            if states[i] != states[i+1]:
                all_transitions[(states[i], states[i+1])] += 1
        
        current_state = states[0]
        current_dwell = 1
        for i in range(1, len(states)):
            if states[i] == current_state:
                current_dwell += 1
            else:
                all_dwell_times.append((current_state, current_dwell))
                current_state = states[i]
                current_dwell = 1
        
        power_ratio = get_power_ratio(eeg, sfreq)
        
        all_ratios.append(ratio)
        all_convergences.append(conv)
        all_states.extend(states)
        all_power_ratios.append(power_ratio)
        
    except Exception as e:
        print(f"S{subj:02d}: Error - {e}")
        continue

all_ratios = np.array(all_ratios)
all_convergences = np.array(all_convergences)
all_power_ratios = np.array(all_power_ratios)

print(f"Analyzed {len(all_ratios)} subjects")

print("\n" + "="*70)
print("TEST 1: THREE ZONES (φ / Shadow / Lock)")
print("="*70)

phi_zone = all_ratios < 1.67
shadow_zone = (all_ratios >= 1.70) & (all_ratios <= 1.83)
lock_zone = all_ratios > 1.86

print(f"\nTHREE ZONES:")
print(f"  φ-zone (<1.67): {phi_zone.sum()} subjects")
print(f"  Shadow (1.70-1.83): {shadow_zone.sum()} subjects")
print(f"  Lock-zone (>1.86): {lock_zone.sum()} subjects")

if phi_zone.any():
    print(f"\nConvergence by zone:")
    print(f"  φ-zone: {all_convergences[phi_zone].mean():.1f}%")
if shadow_zone.any():
    print(f"  Shadow: {all_convergences[shadow_zone].mean():.1f}%")
if lock_zone.any():
    print(f"  Lock: {all_convergences[lock_zone].mean():.1f}%")

print("\n" + "="*70)
print("TEST 2: TEMPORAL SHADOWS (Longest Dwells)")
print("="*70)

all_dwell_times.sort(key=lambda x: x[1], reverse=True)

print("\nTOP 10 LONGEST DWELLS (temporal shadows):")
for state, duration in all_dwell_times[:10]:
    print(f"  V{state+1}: {duration} timepoints")

by_state = {}
for state, dur in all_dwell_times:
    if state not in by_state:
        by_state[state] = []
    by_state[state].append(dur)

print("\nMEAN DWELL TIME BY STATE:")
for state in sorted(by_state.keys()):
    print(f"  V{state+1}: {np.mean(by_state[state]):.2f} ± {np.std(by_state[state]):.2f}")

print("\n" + "="*70)
print("TEST 3: TRANSITION SHADOWS (Never happen)")
print("="*70)

possible = set()
for f in range(7):
    for t in range(7):
        if f != t:
            possible.add((f, t))

actual = set(all_transitions.keys())
shadows = possible - actual

print(f"\nTRANSITION SHADOWS:")
print(f"  Possible transitions: {len(possible)}")
print(f"  Actual transitions: {len(actual)}")
print(f"  Shadow transitions: {len(shadows)}")

if shadows:
    print("\nMISSING TRANSITIONS (SHADOWS):")
    for f, t in sorted(shadows):
        print(f"  V{f+1} → V{t+1}: NEVER HAPPENS")
else:
    print("\n  No pure shadows - all transitions occur at least once!")

least_common = sorted(all_transitions.items(), key=lambda x: x[1])[:5]
print("\nRARELY USED TRANSITIONS (dim shadows):")
for (f, t), count in least_common:
    print(f"  V{f+1} → V{t+1}: only {count} times")

print("\n" + "="*70)
print("TEST 4: V4 NULL ZONE ANALYSIS")
print("="*70)

v4_count = sum(1 for s in all_states if s == 3)
total_states = len(all_states)

print(f"\nV4 (NULL ZONE at 7:4 = 1.75):")
print(f"  Time in V4: {100*v4_count/total_states:.1f}%")

v4_to_phi = sum(all_transitions.get((3, t), 0) for t in [0, 1, 2])
v4_to_lock = sum(all_transitions.get((3, t), 0) for t in [4, 5, 6])
phi_to_v4 = sum(all_transitions.get((f, 3), 0) for f in [0, 1, 2])
lock_to_v4 = sum(all_transitions.get((f, 3), 0) for f in [4, 5, 6])

print(f"\nV4 as GATEWAY:")
print(f"  φ-zone → V4: {phi_to_v4}")
print(f"  V4 → φ-zone: {v4_to_phi}")
print(f"  Lock-zone → V4: {lock_to_v4}")
print(f"  V4 → Lock-zone: {v4_to_lock}")

print(f"\nV4 FLOW BALANCE:")
print(f"  Net flow to φ: {v4_to_phi - phi_to_v4}")
print(f"  Net flow to lock: {v4_to_lock - lock_to_v4}")

print("\n" + "="*70)
print("TEST 5: POWER SHADOWS")
print("="*70)

theta_shadowed = all_power_ratios > 2.0
alpha_shadowed = all_power_ratios < 0.5
balanced = (~theta_shadowed) & (~alpha_shadowed)

print(f"\nPOWER BALANCE:")
print(f"  Theta shadowed (α/θ > 2): {theta_shadowed.sum()} subjects")
print(f"  Alpha shadowed (α/θ < 0.5): {alpha_shadowed.sum()} subjects")
print(f"  Balanced: {balanced.sum()} subjects")

print("\n" + "="*70)
print("CORRELATION: Shadow zone vs Convergence")
print("="*70)

zone_labels = []
for r in all_ratios:
    if r < 1.67:
        zone_labels.append('phi')
    elif r > 1.86:
        zone_labels.append('lock')
    else:
        zone_labels.append('shadow')

phi_subjs = [c for z, c in zip(zone_labels, all_convergences) if z == 'phi']
shadow_subjs = [c for z, c in zip(zone_labels, all_convergences) if z == 'shadow']
lock_subjs = [c for z, c in zip(zone_labels, all_convergences) if z == 'lock']

print(f"\nMean convergence by zone:")
if phi_subjs:
    print(f"  φ-zone: {np.mean(phi_subjs):.1f}% (N={len(phi_subjs)})")
if shadow_subjs:
    print(f"  Shadow: {np.mean(shadow_subjs):.1f}% (N={len(shadow_subjs)})")
if lock_subjs:
    print(f"  Lock: {np.mean(lock_subjs):.1f}% (N={len(lock_subjs)})")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

ax1 = axes[0, 0]
zones = ['φ-zone\n(<1.67)', 'Shadow\n(1.70-1.83)', 'Lock\n(>1.86)']
counts = [phi_zone.sum(), shadow_zone.sum(), lock_zone.sum()]
colors = ['gold', 'gray', 'red']
ax1.bar(zones, counts, color=colors, edgecolor='black')
ax1.set_ylabel('Number of Subjects', fontsize=12)
ax1.set_title('A. Subject Distribution by Zone', fontsize=14)

ax2 = axes[0, 1]
mean_dwells = [np.mean(by_state.get(i, [0])) for i in range(7)]
colors = ['gold' if i <= 2 else 'gray' if i <= 4 else 'red' for i in range(7)]
ax2.bar(range(7), mean_dwells, color=colors, edgecolor='black')
ax2.set_xticks(range(7))
ax2.set_xticklabels(['V1', 'V2\nφ', 'V3', 'V4', 'V5', 'V6', 'V7'])
ax2.set_ylabel('Mean Dwell Time', fontsize=12)
ax2.set_title('B. Mean Dwell Time by State', fontsize=14)

ax3 = axes[0, 2]
state_times = [sum(1 for s in all_states if s == i) for i in range(7)]
ax3.bar(range(7), state_times, color=colors, edgecolor='black')
ax3.set_xticks(range(7))
ax3.set_xticklabels(['V1', 'V2\nφ', 'V3', 'V4', 'V5', 'V6', 'V7'])
ax3.set_ylabel('Total Time', fontsize=12)
ax3.set_title('C. Total Time in Each State', fontsize=14)

ax4 = axes[1, 0]
if phi_subjs:
    ax4.bar(0, np.mean(phi_subjs), color='gold', edgecolor='black', label='φ')
if shadow_subjs:
    ax4.bar(1, np.mean(shadow_subjs), color='gray', edgecolor='black', label='Shadow')
if lock_subjs:
    ax4.bar(2, np.mean(lock_subjs), color='red', edgecolor='black', label='Lock')
ax4.set_xticks([0, 1, 2])
ax4.set_xticklabels(['φ-zone', 'Shadow', 'Lock'])
ax4.set_ylabel('Mean Convergence (%)', fontsize=12)
ax4.set_title('D. Convergence by Zone', fontsize=14)

ax5 = axes[1, 1]
gateway_data = ['φ→V4', 'V4→φ', 'Lock→V4', 'V4→Lock']
gateway_vals = [phi_to_v4, v4_to_phi, lock_to_v4, v4_to_lock]
colors = ['gold', 'lightgreen', 'red', 'orange']
ax5.bar(gateway_data, gateway_vals, color=colors, edgecolor='black')
ax5.set_ylabel('Transition Count', fontsize=12)
ax5.set_title('E. V4 as Gateway', fontsize=14)

ax6 = axes[1, 2]
ax6.axis('off')

summary = f"""
SHADOW DETECTION SUMMARY
================================================

N = {len(all_ratios)} subjects

THREE ZONES:
  φ-zone (<1.67): {phi_zone.sum()} subjects
  Shadow (1.70-1.83): {shadow_zone.sum()} subjects
  Lock (>1.86): {lock_zone.sum()} subjects

TEMPORAL SHADOWS (longest dwells):
"""
for state, dur in all_dwell_times[:3]:
    summary += f"\n  V{state+1}: {dur} timepoints"

summary += f"""

V4 NULL ZONE:
  Time in V4: {100*v4_count/total_states:.1f}%
  Net flow to φ: {v4_to_phi - phi_to_v4}
  Net flow to lock: {v4_to_lock - lock_to_v4}

TRANSITION SHADOWS:
  Pure shadows: {len(shadows)}
  (Transitions that never happen)

INTERPRETATION:
V4 (7:4 = 1.75) acts as the "shadow zone" -
a neutral transition point between the φ
attractor and the 2:1 lock zone.
"""

ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('shadow_detection_analysis.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: shadow_detection_analysis.png")
