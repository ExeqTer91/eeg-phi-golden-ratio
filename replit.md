# EEG Phi Coupling Index Analysis

## Overview

This project investigates **golden ratio (φ ≈ 1.618) organization** in human EEG theta-alpha frequency architecture. The core hypothesis is that the Phi Coupling Index (PCI) predicts theta-alpha convergence across subjects, suggesting mathematical constants like φ and e-1 may characterize brain oscillation dynamics.

**Key metrics computed:**
- Phi Coupling Index (PCI): `log(|ratio - 2.0| / |ratio - φ|)`
- Theta-alpha frequency ratio and convergence percentages
- Cross-frequency coupling and phase synchronization
- Inter-electrode coherence analysis

The project analyzes **320+ subjects** across validated datasets for Frontiers manuscript revision (ID: 1781338).

## Large-Scale Validation Results (February 2026)

### Summary Table

| Dataset | N | Raw r | Raw p | Raw Phi% | FOOOF r | FOOOF p | FOOOF Phi% |
|---------|---|-------|-------|----------|---------|---------|------------|
| **PhysioNet EEGBCI** | 109 | 0.628 | 2.51e-13 | 82.6% | -0.369 | - | 99.1% |
| **MPI-LEMON** | 211 | 0.497 | 1.38e-14 | 79.1% | -0.205 | 2.81e-03 | 97.6% |
| **Combined** | 320 | ~0.55 | <1e-20 | ~80% | - | - | ~98% |

### Key Findings

1. **PCI-Convergence Correlation Replicates**: Significant positive correlation in both datasets (r=0.50-0.63)
2. **Phi-Organization is Robust**: ~80% of subjects show phi-organized ratios (PCI > 0) using raw PSD
3. **1/f Does NOT Explain Effect**: Partial correlation controlling for aperiodic exponent remains strong (r=0.65 vs r=0.63)
4. **FOOOF Correction Increases Phi%**: After removing 1/f, 97-99% of subjects are phi-organized

### Processed Data Files

- `eeg-processing/results/physionet_results.csv` - PhysioNet N=109 raw+FOOOF results
- `eeg-processing/results/physionet_combined_results.csv` - Full analysis with auxiliary metrics
- LEMON results processed on RunPod (N=211)

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Analysis Pipeline

**Signal Processing Approach:**
- Uses scipy.signal.welch for power spectral density estimation
- Band-pass filtering via scipy.signal.butter for theta (4-8 Hz) and alpha (8-13 Hz) extraction
- Spectral centroids computed as power-weighted mean frequencies within bands
- Sliding window analysis (typically 2-second windows with 50% overlap) for time-varying metrics

**Key Computations:**
- PCI formula measures proximity to φ (1.618) vs 2:1 harmonic ratio
- 8 Hz convergence detection identifies when theta and alpha peaks converge
- "Vibraton states" discretize the ratio space into 7 bins for state transition analysis

**Data Processing Libraries:**
- MNE-Python for EEG file I/O (EDF, BDF formats) and preprocessing
- SciPy for signal processing and statistics
- NumPy for numerical operations
- Matplotlib for visualization

### File Organization

| Pattern | Purpose |
|---------|---------|
| `phi_*.py` | Core PCI analysis and phi-specificity tests |
| `*_analysis.py` | Specialized analyses (coherence, cross-frequency, meditation) |
| `split_half_validation.py` | Statistical validation to rule out circularity |
| `app.py` | Streamlit dashboard for results visualization |

### Validation Strategy

Split-half validation separates PCI computation (odd epochs) from convergence measurement (even epochs) to confirm the relationship is a stable trait, not a mathematical artifact.

## External Dependencies

### Python Packages

| Package | Purpose |
|---------|---------|
| `mne` | EEG data loading, preprocessing, and the built-in EEGBCI dataset |
| `scipy` | Signal processing (welch, butter, hilbert), statistics |
| `numpy` | Numerical computation |
| `matplotlib` | Visualization and figure generation |
| `streamlit` | Interactive web dashboard (app.py) |
| `requests` | Downloading external datasets from Zenodo |

### Data Sources

| Source | Access Method |
|--------|---------------|
| **PhysioNet EEGBCI** | Via `mne.datasets.eegbci` - downloads automatically |
| **OpenNeuro ds003969** | BIDS-formatted meditation dataset in `ds003969/` folder |
| **Zenodo Alpha Waves** | Downloaded via HTTP requests to zenodo.org |

### Optional Dependencies

- `fooof` (specparam): For aperiodic-corrected spectral analysis
- `openneuro-py`: Alternative OpenNeuro dataset access

### File Formats

- `.mat` files: MATLAB format spectral data (loaded via scipy.io.loadmat)
- `.edf` files: European Data Format EEG recordings
- `.set` files: EEGLAB format (BIDS datasets)
- BIDS-compliant JSON sidecars for metadata