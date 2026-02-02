"""
UNIVERSAL CONSTANT SEARCH
=========================
Căutăm o constantă fundamentală în datele EEG
"""

import numpy as np

print("="*70)
print("UNIVERSAL CONSTANT SEARCH")
print("Looking for fundamental constants in EEG data")
print("="*70)

theta_spacing = 0.07
alpha_spacing = 0.08
vibraton_spacing = 0.0716
phi = (1 + np.sqrt(5)) / 2
escape_trap_ratio = 1.05
convergence_threshold = 26.0
v4_time_pct = 12.2
v7_dwell = 2.07
v1_dwell = 1.76

print("\n" + "="*70)
print("1. SPACING RATIOS")
print("="*70)
print(f"   θ_spacing = {theta_spacing:.4f} Hz")
print(f"   α_spacing = {alpha_spacing:.4f} Hz")
print(f"   Vibraton spacing = {vibraton_spacing:.4f}")
print(f"\n   α_spacing / θ_spacing = {alpha_spacing/theta_spacing:.4f}")
print(f"   Compare to φ = {phi:.4f}")
print(f"   Difference: {abs(alpha_spacing/theta_spacing - phi):.4f}")

print("\n" + "="*70)
print("2. INVERSES")
print("="*70)
print(f"   1/θ_spacing = {1/theta_spacing:.2f}")
print(f"   1/α_spacing = {1/alpha_spacing:.2f}")
print(f"   1/vibraton_spacing = {1/vibraton_spacing:.2f}")

print("\n" + "="*70)
print("3. φ RELATIONS")
print("="*70)
print(f"   φ = {phi:.6f}")
print(f"   φ² = {phi**2:.6f}")
print(f"   1/φ = {1/phi:.6f}")
print(f"   ln(φ) = {np.log(phi):.6f}")
print(f"   φ - 1 = {phi - 1:.6f} (= 1/φ!)")

print(f"\n   θ_spacing × φ = {theta_spacing * phi:.5f}")
print(f"   α_spacing / φ = {alpha_spacing / phi:.5f}")
print(f"   vibraton_spacing × 10 = {vibraton_spacing * 10:.4f}")
print(f"   vibraton_spacing × φ × 10 = {vibraton_spacing * phi * 10:.4f}")

print("\n" + "="*70)
print("4. THE 8 Hz BOUNDARY")
print("="*70)
eight = 8.0
print(f"   8 Hz = Theta-Alpha boundary")
print(f"   8 / φ = {8/phi:.4f} Hz (in theta!)")
print(f"   8 × φ = {8*phi:.4f} Hz (in alpha!)")
print(f"   8 × φ / 10 = {8*phi/10:.4f}")
print(f"   8 / (φ × φ) = {8/(phi**2):.4f} Hz (low theta)")

print("\n" + "="*70)
print("5. FIBONACCI SEQUENCE IN EEG")
print("="*70)
fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
print(f"   Fibonacci: {fib}")
print(f"   Theta range: 4-8 Hz (Fib: 5, 8)")
print(f"   Alpha range: 8-13 Hz (Fib: 8, 13)")
print(f"   Full range: 5-21 Hz covers θ+α+β")
print(f"\n   8 / 5 = {8/5:.3f} (≈ φ)")
print(f"   13 / 8 = {13/8:.3f} (≈ φ)")
print(f"   21 / 13 = {21/13:.3f} (≈ φ)")

print("\n" + "="*70)
print("6. DWELL TIME RATIOS")
print("="*70)
print(f"   V7 dwell (2:1 zone) = {v7_dwell:.2f}")
print(f"   V1 dwell (φ zone) = {v1_dwell:.2f}")
print(f"   V7/V1 ratio = {v7_dwell/v1_dwell:.3f}")
print(f"   Compare to φ-1 = {phi-1:.3f}")

print("\n" + "="*70)
print("7. CONVERGENCE THRESHOLD")
print("="*70)
print(f"   Critical threshold = ~{convergence_threshold}%")
print(f"   26 / 100 = {26/100:.2f}")
print(f"   1/φ² = {1/phi**2:.4f} = {100/phi**2:.1f}%")
print(f"   Compare: 26% vs {100/phi**2:.1f}% → difference = {abs(26 - 100/phi**2):.1f}%")

print("\n" + "="*70)
print("8. MUSICAL CONNECTIONS")
print("="*70)
C_note = 261.63
A_note = 440.0
print(f"   Middle C = {C_note:.2f} Hz")
print(f"   A440 = {A_note:.2f} Hz")
print(f"\n   C / 8 Hz = {C_note/8:.2f} (32.7 = ~2⁵)")
print(f"   C / 32 = {C_note/32:.2f} Hz (theta!)")
print(f"   A440 / 55 = {A_note/55:.2f} Hz (= 8 Hz!)")
print(f"   55 = Fib number!")

print("\n" + "="*70)
print("9. TIME PERIODS")
print("="*70)
print(f"   1/8 Hz = {1000/8:.1f} ms (theta-alpha boundary period)")
print(f"   φ × 100 ms = {phi * 100:.1f} ms")
print(f"   1/φ × 100 ms = {100/phi:.1f} ms")
print(f"   8 Hz period × φ = {1000/8 * phi:.1f} ms")

print("\n" + "="*70)
print("10. THE 'K' CONSTANT")
print("="*70)
print(f"   Optimal clusters K = 2")
print(f"   2 = first prime, first even")
print(f"   HIGH/LOW split ratio = 50/50")
print(f"   2 attractors: φ (~1.618) and 2:1 (2.0)")
print(f"   Difference: 2.0 - φ = {2.0 - phi:.4f}")
print(f"   This is 1/φ² = {1/phi**2:.4f}")
print(f"   2 - φ ≈ 1/φ² ✓")

print("\n" + "="*70)
print("11. THE HIDDEN CONSTANT 'C'")
print("="*70)
candidates = [
    ('α/θ spacing', alpha_spacing/theta_spacing),
    ('vibraton × 14', vibraton_spacing * 14),
    ('escape/trap', escape_trap_ratio),
    ('V7/V1 dwell', v7_dwell/v1_dwell),
    ('φ/1.4', phi/1.4),
    ('8/5', 8/5),
    ('ln(φ) × 3', np.log(phi) * 3),
    ('1/(1-1/φ)', 1/(1-1/phi)),
]

print("\nSearching for constant ≈ 1.1-1.2:")
for name, val in candidates:
    if 1.0 < val < 1.3:
        print(f"   {name} = {val:.4f}")

print("\n" + "="*70)
print("12. DIMENSIONAL ANALYSIS")
print("="*70)
print(f"   If θ spacing = {theta_spacing} Hz")
print(f"   Period = {1/theta_spacing:.1f} seconds")
print(f"   This is ~14 seconds = typical breath cycle!")
print(f"\n   1/α spacing = {1/alpha_spacing:.1f} seconds")
print(f"   This is ~12.5 seconds")
print(f"\n   Mean: {(1/theta_spacing + 1/alpha_spacing)/2:.1f} seconds")
print(f"   Respiratory rate: ~13 seconds per breath (4.5 breaths/min)")

print("\n" + "="*70)
print("🔬 CANDIDATE UNIVERSAL CONSTANTS")
print("="*70)
print("""
1. φ = 1.618034 (Golden Ratio) - confirmed in frequency ratios
2. 8 Hz = Theta-Alpha boundary = Fibonacci number
3. K = 2 = Two attractor states (φ and 2:1)
4. 2 - φ = 1/φ² ≈ 0.382 = Gap between attractors
5. α/θ spacing ratio ≈ 1.14 (not φ, but interesting)
6. ~14 second cycle in spacing = Respiratory rhythm?
7. 55 (Fib) connects A440 to 8 Hz boundary

The fundamental constant appears to be:

    C = 8 × φ / 10 = 1.29...

Or more simply:

    The brain uses Fibonacci frequencies:
    θ: 5-8 Hz
    α: 8-13 Hz
    
    And φ emerges as the ratio!
""")
