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
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                # Save numerator and denominator to file for Octave
                num_path = tmp / "num.csv"
                den_path = tmp / "den.csv"
                np.savetxt(num_path, np.asarray(num, dtype=float), delimiter=",")
                np.savetxt(den_path, np.asarray(den, dtype=float), delimiter=",")
                proc = subprocess.run([octave, "--quiet", str(script), str(num_path), str(den_path), str(tmp)], check=True, capture_output=True, text=True)
                bode_path = tmp / "bode.csv"
                nyquist_path = tmp / "nyquist.csv"
                step_path = tmp / "step.csv"
                impulse_path = tmp / "impulse.csv"
                root_locus_path = tmp / "root_locus.csv"
                margins_path = tmp / "margins.csv"
                if bode_path.exists():
                    bode_data = np.loadtxt(bode_path, delimiter=",")
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
                    "impulse_available": impulse_path.exists(),
                    "root_locus_available": root_locus_path.exists(),
                    "margins_available": margins_path.exists(),
                }
        except subprocess.SubprocessError:
            pass
    return {
        "backend": "python-fallback",
        "omega": w,
        "mag": mag,
        "phase": phase,
        "nyquist_available": False,
        "step_available": False,
        "impulse_available": False,
        "root_locus_available": False,
        "margins_available": False,
    }
