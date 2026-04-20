# HELIOS

**Author:** George David Tsitlauri  
**Affiliation:** Dept. of Informatics & Telecommunications, University of Thessaly, Greece  
**Contact:** gdtsitlauri@gmail.com  
**Year:** 2026

HELIOS is a multi-language research framework for signal processing, information theory, stochastic systems, and hybrid predictive modeling.

This bootstrap implementation provides:

- A Python-native core that runs today on this machine
- Julia, R, and Octave bridge layers with graceful fallbacks
- A lightweight HELIOS-SPECTRUM pipeline for end-to-end signal analysis
- Baseline result artifacts, tests, and an IEEE-style paper draft
- Extended research utilities for arithmetic coding, Gaussian rate-distortion, Markov hitting/absorption metrics, and continuous-time process simulation

## Layout

- `src/helios/`: Python package
- `julia/`: Julia DSP scripts
- `r/`: R stochastic analysis scripts
- `octave/`: Octave control and signal scripts
- `results/`: Baseline experiment outputs
- `tests/`: Pytest suite
- `paper/`: IEEE-style manuscript draft

## Quick Start

```bash
python3 -m pytest tests/test_helios.py
python3 scripts/run_experiments.py
```

The Python package uses fallbacks automatically when `julia`, `R`, or `octave` are unavailable.

Bridge discovery order:

- Workspace-local runtimes under `.tooling/`
- System-installed `julia`, `R`, and `octave`
- Python fallbacks when the external runtime is not available

## Multi-Language Strategy

- Python orchestrates all workflows and hosts ML models
- Julia scripts target fast FFT, filtering, and wavelet workloads
- R scripts target time-series and stochastic analysis
- Octave scripts target control systems and MATLAB-compatible workflows

## Implemented Research Blocks

- DSP: FFT, STFT, filter design and response, group delay, wavelet decomposition/reconstruction, denoising, Julia bridge
- Information theory: entropy, mutual information, Huffman/LZ78/arithmetic coding, Hamming `(7,4)` and `(15,11)`, convolutional/Viterbi, LDPC-style bit-flip decoding, Reed-Solomon-like BER proxy, turbo-style BER baselines, Gaussian rate-distortion
- Stochastic systems: Markov transition estimation, stationary distributions, hitting times, absorption probabilities, Baum-Welch-style update step, ARIMA/SARIMA-style R forecasting, AR1/AR2 forecasting, Viterbi decoding, Poisson/Brownian/GBM/OU simulation, Kalman tracking, R bridge
- HELIOS-SPECTRUM: wavelet feature extraction, MI-driven selection, temporal prediction, uncertainty estimation, causal graph export
- Octave-compatible control: Bode and Nyquist data export, generated signal CSVs, PNG plot generation, Octave bridge

## Current Environment Notes

- Python is available
- PyTorch with CUDA support is installed
- Julia is installed locally in `.tooling/julia`
- R is installed locally in `.tooling/r-env`
- Octave is installed locally in `.tooling/octave-env`

The bridges are implemented so the architecture is stable now and can switch to the external runtimes once those tools are installed.
## Experiment Results (2026)

The main results of the HELIOS experiments (as generated in the results/ directory) are summarized below:

### 1. FFT Benchmark (Julia vs NumPy)

| Backend | Mean Time (s) | Speedup vs NumPy |
|---------|---------------|------------------|
| NumPy   | 0.00030       | 1.0              |
| Julia   | 0.00011       | 2.25             |

### 2. Channel Coding BER (BER @ SNR=4dB)

| Code                | BER      |
|---------------------|----------|
| Hamming (7,4)       | 0.1128   |
| Hamming (15,11)     | 0.0549   |
| Conv. + Viterbi     | 0.00015  |
| Reed-Solomon-like   | 0.0152   |
| LDPC (bitflip)      | 0.00006  |
| Turbo (simplified)  | 0.000015 |
| Neural Autoencoder  | 0.000037 |

### 3. HELIOS-SPECTRUM vs Baselines (MSE)

| Dataset     | HELIOS-SPECTRUM | FFT+ML | ARIMA | CNN   |
|-------------|-----------------|--------|-------|-------|
| ECG         | 0.0411          | 0.0574 | 0.062 | 0.047 |
| Seismic     | 0.331           | 0.905  | 0.978 | 0.381 |
| Financial   | 0.651           | 4.53   | 4.90  | 0.749 |
| Audio       | 0.248           | 0.787  | 0.850 | 0.285 |

### 4. Stochastic Forecast (indicative)

| Horizon | ARIMA Forecast | SARIMA | GARCH omega | Granger | Cointegration |
|---------|---------------|--------|-------------|---------|---------------|
| 1       | -0.0531       | -0.0547| 0.100       | 0.0366  | -0.991        |
| 2       | -0.0525       | -0.0550| 0.100       | 0.0366  | -0.991        |

### 5. Octave Filter Design (indicative)

| Freq (rad) | Mag1 | Mag2 | Mag3 |
|------------|------|------|------|
| 0.00       | 1.00 | 0.82 | 1.00 |
| 0.12       | 1.00 | 0.82 | 1.00 |

---
For full analysis, see the files in the results/ folder and insert the numbers into the tables of paper/helios_paper.tex.

## Citation

```bibtex
@misc{tsitlauri2026helios,
  author = {George David Tsitlauri},
  title  = {HELIOS: A Multi-Language Framework for Unified Signal Processing, Information Theory, and Stochastic Systems},
  year   = {2026},
  institution = {University of Thessaly},
  email  = {gdtsitlauri@gmail.com}
}
```
