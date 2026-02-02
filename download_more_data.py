"""
DOWNLOAD MORE EEG DATA
======================
Descărcăm cât mai multe date disponibile
"""

import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf
import os
import warnings
warnings.filterwarnings('ignore')

mne.set_log_level('ERROR')

print("="*70)
print("DOWNLOADING MORE EEG DATA")
print("="*70)

print("\n1. PhysioNet Motor Imagery - downloading all 109 subjects (run 1 only)...")

success = 0
failed = []

for subj in range(1, 110):
    try:
        path = eegbci.load_data(subj, [1], update_path=True, verbose=False)
        success += 1
        if subj % 20 == 0:
            print(f"   Downloaded {subj}/109...")
    except Exception as e:
        failed.append(subj)

print(f"\n   SUCCESS: {success} subjects")
print(f"   FAILED: {len(failed)} subjects")

print("\n2. Downloading additional runs for first 60 subjects...")

runs_downloaded = 0
for subj in range(1, 61):
    for run in [2, 3, 4, 5]:
        try:
            path = eegbci.load_data(subj, [run], update_path=True, verbose=False)
            runs_downloaded += 1
        except:
            pass
    if subj % 20 == 0:
        print(f"   Subject {subj}/60 - additional runs...")

print(f"   Additional runs downloaded: {runs_downloaded}")

print("\n3. Checking Sleep PhysioNet...")

try:
    from mne.datasets import sleep_physionet
    
    alice_files = sleep_physionet.age.fetch_data(subjects=[0], recording=[1])
    print(f"   Sleep PhysioNet available! Downloaded sample.")
    
    for subj in range(1, 20):
        try:
            sleep_physionet.age.fetch_data(subjects=[subj], recording=[1])
        except:
            pass
    print(f"   Downloaded up to 20 sleep recordings")
    
except Exception as e:
    print(f"   Sleep PhysioNet not available: {e}")

print("\n" + "="*70)
print("FINAL COUNT")
print("="*70)

import subprocess
result = subprocess.run(['find', '/home/runner/mne_data/', '-name', '*.edf'], 
                       capture_output=True, text=True)
edf_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0

result2 = subprocess.run(['du', '-sh', '/home/runner/mne_data/'], 
                        capture_output=True, text=True)
size = result2.stdout.split()[0] if result2.stdout else "?"

print(f"\nTotal EDF files: {edf_count}")
print(f"Total size: {size}")

unique_subjects = set()
for line in result.stdout.strip().split('\n'):
    if line:
        fname = os.path.basename(line)
        if 'S' in fname:
            subj_id = fname.split('R')[0] if 'R' in fname else fname[:4]
            unique_subjects.add(subj_id)

print(f"Unique subjects (PhysioNet): {len(unique_subjects)}")
