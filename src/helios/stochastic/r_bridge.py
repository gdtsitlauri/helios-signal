from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .core import forecast_ar1
from helios.runtime import resolve_runtime


ROOT = Path(__file__).resolve().parents[3]


def run_arima_bridge(series: np.ndarray, horizon: int = 5) -> dict:
    r_bin = resolve_runtime("R", [".tooling/helios-env/bin/R", ".tooling/r-env/bin/R"])
    if r_bin:
        script = ROOT / "r" / "time_series.R"
        try:
            proc = subprocess.run([r_bin, "--vanilla", "-f", str(script)], check=True, capture_output=True, text=True)
            return {"backend": "R", "stdout": proc.stdout.strip(), "forecast": forecast_ar1(series, horizon)}
        except subprocess.SubprocessError:
            pass
    return {"backend": "python-fallback", "forecast": forecast_ar1(series, horizon)}
