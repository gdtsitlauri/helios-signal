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
            import tempfile
            import pandas as pd
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                input_path = tmp / "signal.csv"
                output_path = tmp / "fft.csv"
                pd.DataFrame({"signal": np.asarray(signal_values, dtype=float)}).to_csv(input_path, index=False)
                proc = subprocess.run(
                    [julia, f"--project={ROOT}", str(script), str(input_path), str(output_path), str(fs)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                if output_path.exists():
                    df = pd.read_csv(output_path)
                    # Expect columns: freqs, spectrum, magnitude
                    result = {col: df[col].to_numpy() for col in df.columns if col in {"freqs", "spectrum", "magnitude"}}
                    return {"backend": "julia", "stdout": proc.stdout.strip(), **result}
        except subprocess.SubprocessError:
            pass
    return {"backend": "python-fallback", **fft(signal_values, fs=fs)}


def filter_via_bridge(signal_values: np.ndarray, fs: float = 1.0, cutoff: float = 5.0) -> dict:
    julia = resolve_runtime("julia", [".tooling/julia/bin/julia", ".tooling/helios-env/bin/julia", ".tooling/julia-env/bin/julia"])
    if julia:
        script = ROOT / "julia" / "filter_design.jl"
        try:
            import tempfile
            import pandas as pd
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                input_path = tmp / "signal.csv"
                output_path = tmp / "filtered.csv"
                pd.DataFrame({"signal": np.asarray(signal_values, dtype=float)}).to_csv(input_path, index=False)
                proc = subprocess.run(
                    [julia, f"--project={ROOT}", str(script), str(input_path), str(output_path), str(fs), str(cutoff)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                if output_path.exists():
                    df = pd.read_csv(output_path)
                    return {"backend": "julia", "stdout": proc.stdout.strip(), "filtered": df["filtered"].to_numpy()}
        except subprocess.SubprocessError:
            pass
    spec = design_filter(cutoff=cutoff, fs=fs, order=4, kind="butter")
    return {
        "backend": "python-fallback",
        "filtered": zero_phase_filter(signal_values, spec),
    }
