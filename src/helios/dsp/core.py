from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import signal


WINDOWS = {
    "hann": np.hanning,
    "hamming": np.hamming,
    "blackman": np.blackman,
}


@dataclass
class FilterSpec:
    b: np.ndarray
    a: np.ndarray
    fs: float
    kind: str
    btype: str


def _as_array(x: Iterable[float]) -> np.ndarray:
    return np.asarray(x, dtype=float)


def fft(x: Iterable[float], fs: float = 1.0, zero_pad: int = 0, window: str | None = None) -> dict:
    samples = _as_array(x)
    n = len(samples) + max(0, int(zero_pad))
    if window:
        samples = samples * WINDOWS[window](len(samples))
    spectrum = np.fft.rfft(samples, n=n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return {"freqs": freqs, "spectrum": spectrum, "magnitude": np.abs(spectrum)}


def ifft(spectrum: Iterable[complex]) -> np.ndarray:
    return np.fft.irfft(np.asarray(spectrum))


def stft(x: Iterable[float], fs: float = 1.0, nperseg: int = 64, noverlap: int = 32, window: str = "hann") -> dict:
    f, t, z = signal.stft(_as_array(x), fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)
    return {"freqs": f, "times": t, "spectrogram": np.abs(z)}


def design_filter(
    cutoff: float | tuple[float, float],
    fs: float,
    order: int = 4,
    kind: str = "butter",
    btype: str = "lowpass",
) -> FilterSpec:
    if kind == "butter":
        b, a = signal.butter(order, cutoff, btype=btype, fs=fs)
    elif kind == "cheby1":
        b, a = signal.cheby1(order, 1, cutoff, btype=btype, fs=fs)
    elif kind == "cheby2":
        b, a = signal.cheby2(order, 20, cutoff, btype=btype, fs=fs)
    elif kind == "ellip":
        b, a = signal.ellip(order, 1, 20, cutoff, btype=btype, fs=fs)
    elif kind == "fir":
        taps = signal.firwin(order + 1, cutoff, fs=fs, pass_zero=btype in {"lowpass", "bandstop"})
        b, a = taps, np.array([1.0])
    else:
        raise ValueError(f"Unsupported filter kind: {kind}")
    return FilterSpec(b=np.asarray(b), a=np.asarray(a), fs=fs, kind=kind, btype=btype)


def frequency_response(spec: FilterSpec, worN: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w, h = signal.freqz(spec.b, spec.a, worN=worN, fs=spec.fs)
    return w, np.abs(h), np.unwrap(np.angle(h))


def apply_filter(x: Iterable[float], spec: FilterSpec) -> np.ndarray:
    return signal.lfilter(spec.b, spec.a, _as_array(x))


def zero_phase_filter(x: Iterable[float], spec: FilterSpec) -> np.ndarray:
    return signal.filtfilt(spec.b, spec.a, _as_array(x))


def group_delay(spec: FilterSpec) -> tuple[np.ndarray, np.ndarray]:
    return signal.group_delay((spec.b, spec.a), fs=spec.fs)


def wavelet_decompose(x: Iterable[float], levels: int = 3) -> list[np.ndarray]:
    coeffs: list[np.ndarray] = []
    current = _as_array(x)
    for _ in range(levels):
        if len(current) % 2:
            current = np.append(current, current[-1])
        approx = (current[0::2] + current[1::2]) / np.sqrt(2.0)
        detail = (current[0::2] - current[1::2]) / np.sqrt(2.0)
        coeffs.append(detail)
        current = approx
    coeffs.append(current)
    return coeffs


def wavelet_reconstruct(coeffs: list[np.ndarray]) -> np.ndarray:
    current = np.asarray(coeffs[-1], dtype=float)
    for detail in reversed(coeffs[:-1]):
        detail = np.asarray(detail, dtype=float)
        up = np.zeros(detail.size * 2, dtype=float)
        up[0::2] = (current + detail) / np.sqrt(2.0)
        up[1::2] = (current - detail) / np.sqrt(2.0)
        current = up
    return current


def wavelet_denoise(x: Iterable[float], levels: int = 3, mode: str = "soft") -> np.ndarray:
    coeffs = wavelet_decompose(x, levels=levels)
    sigma = np.median(np.abs(coeffs[0])) / 0.6745 if coeffs[0].size else 0.0
    threshold = sigma * np.sqrt(2 * np.log(len(_as_array(x)) + 1))
    cleaned = []
    for detail in coeffs[:-1]:
        if mode == "hard":
            cleaned.append(detail * (np.abs(detail) >= threshold))
        else:
            cleaned.append(np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0))
    cleaned.append(coeffs[-1])
    reconstructed = wavelet_reconstruct(cleaned)
    return reconstructed[: len(_as_array(x))]
