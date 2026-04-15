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


def hitting_times(transition: np.ndarray, target_state: int) -> np.ndarray:
    transition = np.asarray(transition, dtype=float)
    n = transition.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[target_state] = False
    q = transition[np.ix_(mask, mask)]
    b = np.ones(q.shape[0])
    h = np.linalg.solve(np.eye(q.shape[0]) - q + 1e-12 * np.eye(q.shape[0]), b)
    out = np.zeros(n, dtype=float)
    out[mask] = h
    return out


def absorption_probabilities(transition: np.ndarray, absorbing_states: list[int]) -> np.ndarray:
    transition = np.asarray(transition, dtype=float)
    n = transition.shape[0]
    absorbing = np.array(sorted(absorbing_states), dtype=int)
    transient = np.array([i for i in range(n) if i not in set(absorbing_states)], dtype=int)
    if len(transient) == 0:
        return np.eye(len(absorbing))
    q = transition[np.ix_(transient, transient)]
    r = transition[np.ix_(transient, absorbing)]
    fundamental = np.linalg.inv(np.eye(len(transient)) - q + 1e-12 * np.eye(len(transient)))
    probs = fundamental @ r
    full = np.zeros((n, len(absorbing)))
    full[transient] = probs
    for idx, state in enumerate(absorbing):
        full[state, idx] = 1.0
    return full


def fit_ar1(series: np.ndarray) -> tuple[float, float]:
    x = np.asarray(series, dtype=float)
    x0, x1 = x[:-1], x[1:]
    phi = float(np.dot(x0, x1) / (np.dot(x0, x0) + 1e-12))
    residuals = x1 - phi * x0
    sigma = float(np.std(residuals))
    return phi, sigma


def fit_ar2(series: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(series, dtype=float)
    target = x[2:]
    design = np.c_[x[1:-1], x[:-2]]
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    residuals = target - design @ coeffs
    return coeffs, float(np.std(residuals))


def forecast_ar1(series: np.ndarray, horizon: int = 5) -> np.ndarray:
    phi, _ = fit_ar1(series)
    preds = []
    current = float(np.asarray(series, dtype=float)[-1])
    for _ in range(horizon):
        current = phi * current
        preds.append(current)
    return np.asarray(preds)


def forecast_ar2(series: np.ndarray, horizon: int = 5) -> np.ndarray:
    coeffs, _ = fit_ar2(series)
    hist = list(np.asarray(series, dtype=float)[-2:])
    preds = []
    for _ in range(horizon):
        nxt = coeffs[0] * hist[-1] + coeffs[1] * hist[-2]
        preds.append(nxt)
        hist.append(float(nxt))
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


def simulate_poisson_process(rate: float, horizon: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    times = []
    t = 0.0
    while t < horizon:
        t += rng.exponential(1.0 / max(rate, 1e-12))
        if t <= horizon:
            times.append(t)
    return np.asarray(times)


def simulate_brownian_motion(steps: int = 100, dt: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    increments = rng.normal(0.0, np.sqrt(dt), size=steps)
    return np.concatenate([[0.0], np.cumsum(increments)])


def simulate_gbm(s0: float = 1.0, mu: float = 0.05, sigma: float = 0.2, steps: int = 100, dt: float = 1 / 252, seed: int = 0) -> np.ndarray:
    w = simulate_brownian_motion(steps=steps, dt=dt, seed=seed)
    t = np.arange(steps + 1) * dt
    return s0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * w)


def simulate_ornstein_uhlenbeck(theta: float = 0.7, mu: float = 0.0, sigma: float = 0.2, steps: int = 100, dt: float = 0.1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(steps + 1)
    for i in range(steps):
        x[i + 1] = x[i] + theta * (mu - x[i]) * dt + sigma * np.sqrt(dt) * rng.normal()
    return x


def viterbi_decode(observations: np.ndarray, transition: np.ndarray, emission: np.ndarray, initial: np.ndarray | None = None) -> np.ndarray:
    obs = np.asarray(observations, dtype=int)
    transition = np.asarray(transition, dtype=float)
    emission = np.asarray(emission, dtype=float)
    n_states = transition.shape[0]
    initial = np.ones(n_states) / n_states if initial is None else np.asarray(initial, dtype=float)
    log_delta = np.log(initial + 1e-12) + np.log(emission[:, obs[0]] + 1e-12)
    psi = np.zeros((len(obs), n_states), dtype=int)
    for t in range(1, len(obs)):
        scores = log_delta[:, None] + np.log(transition + 1e-12)
        psi[t] = np.argmax(scores, axis=0)
        log_delta = np.max(scores, axis=0) + np.log(emission[:, obs[t]] + 1e-12)
    states = np.zeros(len(obs), dtype=int)
    states[-1] = int(np.argmax(log_delta))
    for t in range(len(obs) - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states
