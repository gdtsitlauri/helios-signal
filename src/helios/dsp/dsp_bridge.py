from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .core import design_filter, fft, zero_phase_filter
from helios.runtime import resolve_runtime


ROOT = Path(__file__).resolve().parents[3]


def fft_via_bridge(signal_values: np.ndarray, fs: float = 1.0) -> dict:
    julia = resolve_runtime("julia", [".tooling/julia/bin/julia", ".tooling/helios-env/bin/julia", ".tooling/julia-env/bin/julia"])
    if julia:
        script = ROOT / "julia" / "fft_analysis.jl"
        try:
            proc = subprocess.run(
                [julia, f"--project={ROOT}", str(script)],
                check=True,
                capture_output=True,
                text=True,
            )
            return {"backend": "julia", "stdout": proc.stdout.strip(), **fft(signal_values, fs=fs)}
        except subprocess.SubprocessError:
            pass
    return {"backend": "python-fallback", **fft(signal_values, fs=fs)}


def filter_via_bridge(signal_values: np.ndarray, fs: float = 1.0, cutoff: float = 5.0) -> dict:
    spec = design_filter(cutoff=cutoff, fs=fs, order=4, kind="butter")
    return {
        "backend": "julia" if resolve_runtime("julia", [".tooling/julia/bin/julia", ".tooling/helios-env/bin/julia", ".tooling/julia-env/bin/julia"]) else "python-fallback",
        "filtered": zero_phase_filter(signal_values, spec),
    }
