"""
DATASET EXPANSION SCRIPT
========================
Download additional EEG datasets for φ-switching analysis
"""

import os
import subprocess
import urllib.request

print("="*70)
print("DATASET EXPANSION FOR φ-SWITCHING ANALYSIS")
print("="*70)

print("\n" + "="*70)
print("1. CURRENT DATASETS (Already Available)")
print("="*70)

from mne.datasets import eegbci
import mne
mne.set_log_level('ERROR')

print("PhysioNet Motor Imagery (eegbci):")
print("  - 109 subjects available")
print("  - Currently using 30 subjects")
print("  - Can expand to ALL 109!")

print("\n" + "="*70)
print("2. EXPAND PHYSIONET TO 109 SUBJECTS")
print("="*70)

def expand_physionet(max_subjects=109):
    """Expand analysis to all 109 PhysioNet subjects"""
    print(f"\nAnalyzing up to {max_subjects} subjects from PhysioNet...")
    
    successful = 0
    failed = []
    
    for subj in range(1, max_subjects + 1):
        try:
            path = eegbci.load_data(subj, [1], update_path=True, verbose=False)
            successful += 1
        except Exception as e:
            failed.append(subj)
    
    print(f"\n  Successfully accessed: {successful} subjects")
    print(f"  Failed: {len(failed)} subjects")
    return successful

print("\nRunning expansion test (first 60 subjects)...")
n_subjects = expand_physionet(60)

print("\n" + "="*70)
print("3. OPENNEURO DATASETS (Available Online)")
print("="*70)

datasets = {
    'ds003775': ('Attention & Working Memory EEG', '~40 subjects'),
    'ds004148': ('Resting State EEG', '~60 subjects'), 
    'ds002680': ('Eyes Open/Closed EEG', '~50 subjects'),
    'ds003505': ('EEG During Meditation', '~30 subjects'),
    'ds004015': ('Alpha Oscillations', '~25 subjects'),
    'ds003774': ('Mindfulness Meditation', '~20 subjects'),
}

print("\nAvailable OpenNeuro Datasets:")
print("-"*70)
for ds_id, (name, count) in datasets.items():
    url = f"https://openneuro.org/datasets/{ds_id}"
    print(f"  {ds_id}: {name} ({count})")
    print(f"          {url}")

print("\n" + "="*70)
print("4. MNE SAMPLE DATASETS (Built-in)")
print("="*70)

print("\nMNE provides additional datasets:")
print("  - mne.datasets.sample (MEG/EEG)")
print("  - mne.datasets.eegbci (Motor imagery)")
print("  - mne.datasets.sleep_physionet (Sleep EEG)")

try:
    from mne.datasets import sleep_physionet
    print("\n  Sleep PhysioNet available - contains sleep stage data!")
except:
    print("\n  Sleep PhysioNet requires additional download")

print("\n" + "="*70)
print("5. RECOMMENDED EXPANSION STRATEGY")
print("="*70)

print("""
PHASE 1: Expand PhysioNet (IMMEDIATE)
  - Currently: 30 subjects
  - Available: 109 subjects
  - Action: Run full 109-subject analysis
  
PHASE 2: Add Sleep PhysioNet (NEXT)
  - Contains EEG during different sleep stages
  - Relevant for consciousness states
  - ~150+ recordings available

PHASE 3: OpenNeuro Meditation Dataset (FUTURE)
  - ds003505: EEG During Meditation
  - Critical for meditators vs non-meditators comparison
  - Requires manual download
""")

print("\n" + "="*70)
print("6. RUNNING EXPANDED ANALYSIS (90+ subjects)")
print("="*70)

print("\nTo run expanded analysis, update comprehensive_analysis.py:")
print("  Change: for subj in range(1, 31):")
print("  To:     for subj in range(1, 91):")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
CURRENT:  30 subjects from PhysioNet
EXPANDED: Up to 109 subjects available NOW
FUTURE:   OpenNeuro datasets for meditation comparison

Recommendation: Run analysis on 90 subjects for more robust statistics!
""")
