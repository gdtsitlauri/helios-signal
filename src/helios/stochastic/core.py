from __future__ import annotations

import numpy as np


def estimate_transition_matrix(states: np.ndarray, n_states: int | None = None) -> np.ndarray:
    states = np.asarray(states, dtype=int)
    n = int(states.max()) + 1 if n_states is None else n_states
    counts = np.zeros((n, n), dtype=float)
    for a, b in zip(states[:-1], states[1:]):
        counts[a, b] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
    return probs


def stationary_distribution(transition: np.ndarray, steps: int = 256) -> np.ndarray:
    n = transition.shape[0]
    dist = np.ones(n) / n
    for _ in range(steps):
        dist = dist @ transition
    total = dist.sum()
    return dist / total if total else np.ones(n) / n


def fit_ar1(series: np.ndarray) -> tuple[float, float]:
    x = np.asarray(series, dtype=float)
    x0, x1 = x[:-1], x[1:]
    phi = float(np.dot(x0, x1) / (np.dot(x0, x0) + 1e-12))
    residuals = x1 - phi * x0
    sigma = float(np.std(residuals))
    return phi, sigma


def forecast_ar1(series: np.ndarray, horizon: int = 5) -> np.ndarray:
    phi, _ = fit_ar1(series)
    preds = []
    current = float(np.asarray(series, dtype=float)[-1])
    for _ in range(horizon):
        current = phi * current
        preds.append(current)
    return np.asarray(preds)


def granger_score(x: np.ndarray, y: np.ndarray, lag: int = 1) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) <= lag + 1 or len(y) <= lag + 1:
        return 0.0
    target = y[lag:]
    y_lag = y[:-lag]
    x_lag = x[:-lag]
    baseline = np.c_[np.ones_like(y_lag), y_lag]
    full = np.c_[np.ones_like(y_lag), y_lag, x_lag]
    beta_base, *_ = np.linalg.lstsq(baseline, target, rcond=None)
    beta_full, *_ = np.linalg.lstsq(full, target, rcond=None)
    rss_base = np.mean((target - baseline @ beta_base) ** 2)
    rss_full = np.mean((target - full @ beta_full) ** 2)
    return float(max(rss_base - rss_full, 0.0) / (rss_base + 1e-12))
