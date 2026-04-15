from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from scipy import signal
from helios.runtime import resolve_runtime


ROOT = Path(__file__).resolve().parents[3]


def run_bode_bridge(num=(1.0,), den=(1.0, 1.0)) -> dict:
    octave = resolve_runtime("octave", [".tooling/helios-env/bin/octave", ".tooling/octave-env/bin/octave"])
    w = np.logspace(-2, 2, 128)
    _, mag, phase = signal.bode((num, den), w=w)
    if octave:
        script = ROOT / "octave" / "control_systems.m"
        try:
            proc = subprocess.run([octave, "--quiet", str(script)], check=True, capture_output=True, text=True)
            return {"backend": "octave", "stdout": proc.stdout.strip(), "omega": w, "mag": mag, "phase": phase}
        except subprocess.SubprocessError:
            pass
    return {"backend": "python-fallback", "omega": w, "mag": mag, "phase": phase}
