# HELIOS

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
