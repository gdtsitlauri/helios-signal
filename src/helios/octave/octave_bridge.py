from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy import signal
from helios.runtime import resolve_runtime


ROOT = Path(__file__).resolve().parents[3]


def run_bode_bridge(num=(1.0,), den=(1.0, 1.0)) -> dict:
    octave = resolve_runtime("octave", [".tooling/octave-env/bin/octave", ".tooling/helios-env/bin/octave"])
    w = np.logspace(-2, 2, 128)
    _, mag, phase = signal.bode((num, den), w=w)
    if octave:
        script = ROOT / "octave" / "control_systems.m"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                proc = subprocess.run([octave, "--quiet", str(script), tmpdir], check=True, capture_output=True, text=True)
                bode_path = Path(tmpdir) / "bode.csv"
                nyquist_path = Path(tmpdir) / "nyquist.csv"
                step_path = Path(tmpdir) / "step.csv"
                if bode_path.exists():
                    bode_data = np.loadtxt(bode_path)
                    w = bode_data[:, 0]
                    mag = bode_data[:, 1]
                    phase = bode_data[:, 2]
                return {
                    "backend": "octave",
                    "stdout": proc.stdout.strip(),
                    "omega": w,
                    "mag": mag,
                    "phase": phase,
                    "nyquist_available": nyquist_path.exists(),
                    "step_available": step_path.exists(),
                }
        except subprocess.SubprocessError:
            pass
    return {"backend": "python-fallback", "omega": w, "mag": mag, "phase": phase, "nyquist_available": False, "step_available": False}
