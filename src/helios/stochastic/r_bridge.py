from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .core import forecast_ar1
from helios.runtime import resolve_runtime


ROOT = Path(__file__).resolve().parents[3]


def run_arima_bridge(series: np.ndarray, horizon: int = 5) -> dict:
    r_bin = resolve_runtime("R", [".tooling/r-env/bin/R", ".tooling/helios-env/bin/R"])
    if r_bin:
        script = ROOT / "r" / "time_series.R"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                input_path = tmp / "series.csv"
                output_path = tmp / "forecast.csv"
                import pandas as pd
                values = np.asarray(series, dtype=float)
                aux = np.roll(values, 1)
                pd.DataFrame({"signal": values, "aux": aux}).to_csv(input_path, index=False)
                proc = subprocess.run(
                    [r_bin, "--vanilla", "-f", str(script), "--args", str(input_path), str(output_path), str(horizon)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                if output_path.exists():
                    frame = pd.read_csv(output_path)
                    return {
                        "backend": "R",
                        "stdout": proc.stdout.strip(),
                        "forecast": frame["forecast_arima"].to_numpy(),
                        "details": frame.to_dict(orient="list"),
                    }
        except subprocess.SubprocessError:
            pass
    return {"backend": "python-fallback", "forecast": forecast_ar1(series, horizon)}
