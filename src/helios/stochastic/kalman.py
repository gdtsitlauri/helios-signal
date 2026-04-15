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
