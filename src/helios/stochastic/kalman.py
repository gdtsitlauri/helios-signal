from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanFilter:
    F: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    x: np.ndarray
    P: np.ndarray

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> None:
        y = z - self.H @ self.x
        s = self.H @ self.P @ self.H.T + self.R
        k = self.P @ self.H.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        i = np.eye(self.P.shape[0])
        self.P = (i - k @ self.H) @ self.P


@dataclass
class ExtendedKalmanFilter:
    f: callable
    h: callable
    jf: callable
    jh: callable
    Q: np.ndarray
    R: np.ndarray
    x: np.ndarray
    P: np.ndarray

    def predict(self, u: np.ndarray | None = None) -> None:
        self.x = np.asarray(self.f(self.x, u), dtype=float)
        F = np.asarray(self.jf(self.x, u), dtype=float)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z: np.ndarray) -> None:
        H = np.asarray(self.jh(self.x), dtype=float)
        y = np.asarray(z, dtype=float) - np.asarray(self.h(self.x), dtype=float)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = self.P - K @ H @ self.P


@dataclass
class UnscentedKalmanFilter:
    f: callable
    h: callable
    Q: np.ndarray
    R: np.ndarray
    x: np.ndarray
    P: np.ndarray
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0

    def _sigma_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(self.x)
        lam = self.alpha**2 * (n + self.kappa) - n
        scale = n + lam
        chol = np.linalg.cholesky(scale * self.P + 1e-9 * np.eye(n))
        points = [self.x]
        for i in range(n):
            points.append(self.x + chol[:, i])
            points.append(self.x - chol[:, i])
        wm = np.full(2 * n + 1, 1 / (2 * scale))
        wc = np.full(2 * n + 1, 1 / (2 * scale))
        wm[0] = lam / scale
        wc[0] = lam / scale + (1 - self.alpha**2 + self.beta)
        return np.asarray(points), wm, wc

    def predict(self, u: np.ndarray | None = None) -> None:
        sigma, wm, wc = self._sigma_points()
        propagated = np.asarray([self.f(pt, u) for pt in sigma], dtype=float)
        self.x = np.sum(wm[:, None] * propagated, axis=0)
        diff = propagated - self.x
        self.P = sum(wc[i] * np.outer(diff[i], diff[i]) for i in range(len(wc))) + self.Q

    def update(self, z: np.ndarray) -> None:
        sigma, wm, wc = self._sigma_points()
        obs_points = np.asarray([self.h(pt) for pt in sigma], dtype=float)
        z_pred = np.sum(wm[:, None] * obs_points, axis=0)
        z_diff = obs_points - z_pred
        x_diff = sigma - self.x
        S = sum(wc[i] * np.outer(z_diff[i], z_diff[i]) for i in range(len(wc))) + self.R
        Cxz = sum(wc[i] * np.outer(x_diff[i], z_diff[i]) for i in range(len(wc)))
        K = Cxz @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(z, dtype=float) - z_pred)
        self.P = self.P - K @ S @ K.T


def run_tracking_demo(steps: int = 40, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    true_positions = np.linspace(0, 10, steps)
    measurements = true_positions + rng.normal(0, 0.6, size=steps)
    kf = KalmanFilter(
        F=np.array([[1.0, 1.0], [0.0, 1.0]]),
        H=np.array([[1.0, 0.0]]),
        Q=np.array([[0.01, 0.0], [0.0, 0.03]]),
        R=np.array([[0.36]]),
        x=np.array([0.0, 0.0]),
        P=np.eye(2),
    )
    estimates = []
    for z in measurements:
        kf.predict()
        kf.update(np.array([z]))
        estimates.append(kf.x[0])
    estimates = np.asarray(estimates)
    return {
        "truth": true_positions,
        "measurements": measurements,
        "estimates": estimates,
        "measurement_rmse": float(np.sqrt(np.mean((measurements - true_positions) ** 2))),
        "estimate_rmse": float(np.sqrt(np.mean((estimates - true_positions) ** 2))),
    }


def run_nonlinear_tracking_demo(steps: int = 40, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    truth = np.linspace(0.2, 8.0, steps)
    measurements = truth**2 + rng.normal(0, 1.0, size=steps)
    drift = float(truth[1] - truth[0])

    def f(x, _u):
        return np.array([x[0] + drift])

    def h(x):
        return np.array([x[0] ** 2])

    def jf(_x, _u):
        return np.array([[1.0]])

    def jh(x):
        return np.array([[2.0 * x[0]]])

    ekf = ExtendedKalmanFilter(f=f, h=h, jf=jf, jh=jh, Q=np.array([[0.005]]), R=np.array([[1.0]]), x=np.array([0.2]), P=np.eye(1))
    ukf = UnscentedKalmanFilter(f=f, h=h, Q=np.array([[0.005]]), R=np.array([[1.0]]), x=np.array([0.2]), P=np.eye(1))

    ekf_est, ukf_est = [], []
    for z in measurements:
        ekf.predict()
        ekf.update(np.array([z]))
        ukf.predict()
        ukf.update(np.array([z]))
        ekf_est.append(float(ekf.x[0]))
        ukf_est.append(float(ukf.x[0]))

    ekf_est = np.asarray(ekf_est)
    ukf_est = np.asarray(ukf_est)
    return {
        "truth": truth,
        "measurements": measurements,
        "ekf_estimates": ekf_est,
        "ukf_estimates": ukf_est,
        "measurement_rmse": float(np.sqrt(np.mean((measurements - truth**2) ** 2))),
        "measurement_state_rmse": float(np.sqrt(np.mean((np.sqrt(np.maximum(measurements, 0)) - truth) ** 2))),
        "ekf_rmse": float(np.sqrt(np.mean((ekf_est - truth) ** 2))),
        "ukf_rmse": float(np.sqrt(np.mean((ukf_est - truth) ** 2))),
    }
