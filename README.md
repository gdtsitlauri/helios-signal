# HELIOS

HELIOS is a multi-language research framework for signal processing, information theory, stochastic systems, and hybrid predictive modeling.

This bootstrap implementation provides:

- A Python-native core that runs today on this machine
- Julia, R, and Octave bridge layers with graceful fallbacks
- A lightweight HELIOS-SPECTRUM pipeline for end-to-end signal analysis
- Baseline result artifacts, tests, and an IEEE-style paper draft

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

The Python package uses fallbacks when `julia`, `R`, or `octave` are unavailable, which is the current state of this workspace.

Bridge discovery order:

- Workspace-local runtimes under `.tooling/`
- System-installed `julia`, `R`, and `octave`
- Python fallbacks when the external runtime is not available

## Multi-Language Strategy

- Python orchestrates all workflows and hosts ML models
- Julia scripts target fast FFT, filtering, and wavelet workloads
- R scripts target time-series and stochastic analysis
- Octave scripts target control systems and MATLAB-compatible workflows

## Current Environment Notes

- Python is available
- PyTorch with CUDA support is installed
- Julia, R, and Octave are not currently installed in this workspace

The bridges are implemented so the architecture is stable now and can switch to the external runtimes once those tools are installed.
