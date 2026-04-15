from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from helios.dsp.core import wavelet_decompose
from helios.information_theory.core import mutual_information
from helios.stochastic.core import granger_score


class TinyTCN(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)[..., -1]


def _subband_features(signal: np.ndarray) -> list[np.ndarray]:
    coeffs = wavelet_decompose(signal, levels=3)
    return [np.abs(c) for c in coeffs[:-1]] + [coeffs[-1]]


def _score_features(features: list[np.ndarray], target: np.ndarray) -> list[tuple[int, float]]:
    scored = []
    target_bins = np.digitize(target, np.histogram_bin_edges(target, bins=8))
    for idx, feat in enumerate(features):
        bins = np.digitize(feat[: len(target)], np.histogram_bin_edges(feat[: len(target)], bins=8))
        scored.append((idx, mutual_information(bins.tolist(), target_bins.tolist())))
    return sorted(scored, key=lambda item: item[1], reverse=True)


def _build_sequences(features: list[np.ndarray], target: np.ndarray, seq_len: int = 8) -> tuple[np.ndarray, np.ndarray]:
    min_len = min(min(len(f) for f in features), len(target))
    stacked = np.vstack([f[:min_len] for f in features])
    xs, ys = [], []
    for i in range(seq_len, min_len):
        xs.append(stacked[:, i - seq_len : i])
        ys.append(target[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def run_helios_spectrum(signal: np.ndarray, target: np.ndarray, output_dir: str | Path | None = None, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    features = _subband_features(np.asarray(signal, dtype=float))
    ranked = _score_features(features, np.asarray(target, dtype=float))
    selected_idx = [idx for idx, _ in ranked[:2]]
    selected = [features[idx] for idx in selected_idx]

    xs, ys = _build_sequences(selected, np.asarray(target, dtype=float))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyTCN(in_channels=len(selected)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    x_tensor = torch.tensor(xs, device=device)
    y_tensor = torch.tensor(ys[:, None], device=device)
    for _ in range(25):
        optimizer.zero_grad()
        pred = model(x_tensor)
        loss = loss_fn(pred, y_tensor)
        loss.backward()
        optimizer.step()

    flat_x = xs.reshape(xs.shape[0], -1)
    linear_design = np.c_[np.ones(len(flat_x)), flat_x]
    linear_beta, *_ = np.linalg.lstsq(linear_design, ys, rcond=None)
    linear_preds = linear_design @ linear_beta

    model.train()
    samples = []
    with torch.no_grad():
        for _ in range(8):
            samples.append(model(x_tensor).cpu().numpy().ravel())
    neural_preds = np.mean(samples, axis=0)
    preds = 0.35 * neural_preds + 0.65 * linear_preds
    uncertainty = np.std(samples, axis=0)

    causal_graph = {}
    for i in range(len(selected)):
        for j in range(len(selected)):
            if i != j:
                causal_graph[f"subband_{i}->subband_{j}"] = granger_score(selected[i], selected[j], lag=1)

    baseline = float(np.mean((ys - np.mean(ys)) ** 2))
    mse = float(np.mean((ys - preds) ** 2))
    summary = {
        "selected_subbands": selected_idx,
        "feature_scores": ranked,
        "mse": mse,
        "baseline_mse": baseline,
        "improvement": baseline - mse,
        "uncertainty_mean": float(np.mean(uncertainty)),
        "device": str(device),
        "causal_graph": causal_graph,
        "regime_changes": int(np.sum(np.abs(np.diff(np.signbit(np.diff(signal)))))),
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "comparison_table.csv").write_text(
            "model,mse\nHELIOS-SPECTRUM,{:.6f}\nMeanBaseline,{:.6f}\n".format(mse, baseline),
            encoding="utf-8",
        )
        (out / "causal_graph.json").write_text(json.dumps(causal_graph, indent=2), encoding="utf-8")
    return summary
