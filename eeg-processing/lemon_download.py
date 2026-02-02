#!/usr/bin/env python3
"""LEMON Dataset Download with Retry"""
import os
import urllib.request
import re
import time
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path('/workspace/lemon_data')
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = 'https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID/'

def get_subject_list():
    """Get list of subjects from FTP"""
    print("Getting subject list...")
    with urllib.request.urlopen(BASE_URL, timeout=60) as response:
        html = response.read().decode()
    subjects = re.findall(r'href="(sub-\d+)/"', html)
    print(f"Found {len(subjects)} subjects")
    return sorted(subjects)

def download_subject(subj, max_retries=3):
    """Download all files for one subject with retry"""
    subj_dir = DATA_DIR / subj
    subj_dir.mkdir(exist_ok=True)
    
    for ext in ['.vhdr', '.vmrk', '.eeg']:
        filename = f'{subj}{ext}'
        url = f'{BASE_URL}{subj}/RSEEG/{filename}'
        local_path = subj_dir / filename
        
        if local_path.exists() and local_path.stat().st_size > 0:
            continue
        
        for attempt in range(max_retries):
            try:
                urllib.request.urlretrieve(url, local_path)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"Failed: {subj} - {e}")
                    return False
    return True

def main():
    subjects = get_subject_list()
    
    success = 0
    for subj in tqdm(subjects, desc="Downloading LEMON"):
        if download_subject(subj):
            success += 1
    
    print(f"\nDownloaded: {success}/{len(subjects)}")
    
    # Save subject list
    with open(DATA_DIR / 'subjects.txt', 'w') as f:
        for subj in subjects:
            f.write(subj + '\n')

if __name__ == '__main__':
    main()
