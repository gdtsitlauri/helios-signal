from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from helios.dsp import (
    dft,
    design_filter,
    fft,
    spectrogram,
    wavelet_decompose,
    wavelet_packet_decompose,
    wavelet_reconstruct,
    zero_phase_filter,
)
from helios.dsp.dsp_bridge import fft_via_bridge
from helios.helios_spectrum import run_helios_spectrum
from helios.information_theory import (
    arithmetic_encode,
    convolutional_encode,
    hamming1511_decode,
    hamming1511_encode,
    huffman_compress,
    ldpc_bitflip_decode,
    ldpc_encode_small,
    lz78_compress,
    rate_distortion_gaussian,
    turbo_code_ber,
    viterbi_decode_convolutional,
)
from helios.octave import run_bode_bridge
from helios.stochastic import (
    absorption_probabilities,
    estimate_transition_matrix,
    forecast_ar2,
    hitting_times,
    run_arima_bridge,
    run_tracking_demo,
    run_nonlinear_tracking_demo,
    simulate_gbm,
    simulate_ornstein_uhlenbeck,
    simulate_poisson_process,
    stationary_distribution,
    viterbi_decode,
)


def test_fft_correctness():
    fs = 128
    t = np.arange(fs) / fs
    x = np.sin(2 * np.pi * 8 * t)
    result = fft(x, fs=fs)
    peak = result["freqs"][np.argmax(result["magnitude"])]
    assert abs(peak - 8) < 1e-6
    manual = dft(x[:16])
    assert np.argmax(np.abs(manual)) in {1, 15}


def test_filter_design():
    fs = 100
    t = np.arange(0, 1, 1 / fs)
    x = np.sin(2 * np.pi * 2 * t) + 0.7 * np.sin(2 * np.pi * 20 * t)
    spec = design_filter(cutoff=5, fs=fs, order=4)
    y = zero_phase_filter(x, spec)
    before = fft(x, fs=fs)["magnitude"]
    after = fft(y, fs=fs)["magnitude"]
    freqs = fft(x, fs=fs)["freqs"]
    idx20 = np.argmin(np.abs(freqs - 20))
    assert after[idx20] < before[idx20] * 0.5


def test_wavelet_reconstruction():
    x = np.linspace(0, 1, 64)
    coeffs = wavelet_decompose(x, levels=3)
    recon = wavelet_reconstruct(coeffs)[: len(x)]
    assert np.allclose(x, recon, atol=1e-8)
    packets = wavelet_packet_decompose(x, levels=2)
    assert len(packets) == 4


def test_spectrogram_shape():
    x = np.sin(np.linspace(0, 8 * np.pi, 128))
    spec = spectrogram(x, fs=128, nperseg=32, noverlap=16)
    assert spec["power"].ndim == 2


def test_huffman_coding():
    text = "aaaaabbbbcccdde"
    compressed = huffman_compress(text)
    assert compressed["compression_ratio"] > 1.0


def test_compression_and_rate_distortion_extensions():
    text = "signal signal signal processing"
    arithmetic = arithmetic_encode(text)
    lz = lz78_compress(text)
    rd = rate_distortion_gaussian(variance=1.0, distortions=np.array([0.1, 0.2, 0.5]))
    assert arithmetic["compression_ratio"] > 1.0
    assert lz["compression_ratio"] > 0.8
    assert np.all(np.diff(rd) <= 1e-9)


def test_extended_channel_codes():
    bits11 = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    enc1511 = hamming1511_encode(bits11)
    noisy1511 = enc1511.copy()
    noisy1511[3] ^= 1
    dec1511 = hamming1511_decode(noisy1511)
    assert np.array_equal(dec1511[:11], bits11)

    bits = np.array([1, 0, 1, 1, 0, 0])
    conv = convolutional_encode(bits)
    dec = viterbi_decode_convolutional(conv)
    assert np.array_equal(dec[: len(bits)], bits)

    ldpc_msg = np.array([1, 0, 1])
    ldpc_code = ldpc_encode_small(ldpc_msg)
    ldpc_noisy = ldpc_code.copy()
    ldpc_noisy[3] ^= 1
    ldpc_dec = ldpc_bitflip_decode(ldpc_noisy)
    assert np.array_equal(ldpc_dec[:3], ldpc_msg)


def test_turbo_code_ber():
    assert turbo_code_ber(4.0) < 0.01


def test_markov_stationary():
    states = np.array([0, 1, 2, 1, 0, 1, 2, 2, 1, 0])
    transition = estimate_transition_matrix(states, n_states=3)
    stationary = stationary_distribution(transition)
    assert np.isclose(stationary.sum(), 1.0)


def test_stochastic_extensions():
    transition = np.array([[0.5, 0.5, 0.0], [0.2, 0.5, 0.3], [0.0, 0.0, 1.0]])
    hits = hitting_times(transition, target_state=2)
    absorb = absorption_probabilities(transition, absorbing_states=[2])
    assert hits[2] == 0.0
    assert np.allclose(absorb[2], np.array([1.0]))

    series = np.array([1.0, 0.9, 0.82, 0.77, 0.70, 0.65])
    ar2 = forecast_ar2(series, horizon=3)
    assert len(ar2) == 3

    pois = simulate_poisson_process(rate=2.0, horizon=3.0, seed=0)
    gbm = simulate_gbm(steps=32, seed=0)
    ou = simulate_ornstein_uhlenbeck(steps=32, seed=0)
    assert np.all(np.diff(pois) >= 0)
    assert gbm.shape == (33,)
    assert ou.shape == (33,)

    decoded = viterbi_decode(
        np.array([0, 0, 1, 1, 1, 0]),
        transition=np.array([[0.85, 0.15], [0.2, 0.8]]),
        emission=np.array([[0.9, 0.1], [0.2, 0.8]]),
    )
    assert decoded.shape == (6,)


def test_kalman_convergence():
    result = run_tracking_demo()
    assert result["estimate_rmse"] < result["measurement_rmse"]


def test_nonlinear_kalman_convergence():
    result = run_nonlinear_tracking_demo()
    assert result["ekf_rmse"] < result["measurement_state_rmse"]
    assert result["ukf_rmse"] < result["measurement_state_rmse"]


def test_helios_spectrum_pipeline(tmp_path: Path):
    t = np.linspace(0, 8 * np.pi, 256)
    signal = np.sin(t) + 0.2 * np.sin(4 * t)
    target = np.roll(signal, -1)
    result = run_helios_spectrum(signal, target, output_dir=tmp_path)
    assert result["mse"] < result["baseline_mse"]
    assert (tmp_path / "comparison_table.csv").exists()
    assert (tmp_path / "causal_graph.json").exists()


def test_julia_bridge():
    x = np.sin(np.linspace(0, 2 * np.pi, 64))
    result = fft_via_bridge(x, fs=64)
    assert result["backend"] in {"julia", "python-fallback"}
    assert len(result["magnitude"]) > 0


def test_r_bridge():
    x = np.cos(np.linspace(0, 2 * np.pi, 32))
    result = run_arima_bridge(x)
    assert result["backend"] in {"R", "python-fallback"}
    assert len(result["forecast"]) == 5


def test_octave_bridge():
    result = run_bode_bridge()
    assert result["backend"] in {"octave", "python-fallback"}
    assert len(result["omega"]) == len(result["mag"]) == len(result["phase"])
    assert "nyquist_available" in result


def test_packaged_results_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "results" / "helios_spectrum" / "comparison_table.csv").exists()
    graph = json.loads((root / "results" / "helios_spectrum" / "causal_graph.json").read_text())
    assert graph
    assert (root / "results" / "information_theory" / "rate_distortion.csv").exists()
    assert (root / "results" / "stochastic" / "continuous_processes.csv").exists()
    comp = root / "results" / "helios_spectrum" / "comparison_table.csv"
    assert "dataset" in comp.read_text().splitlines()[0]



    import subprocess
    import pandas as pd
    tmp_dir = Path("/tmp/helios_r_arima")
    tmp_dir.mkdir(exist_ok=True)
    input_csv = tmp_dir / "series.csv"
    output_csv = tmp_dir / "arima.csv"
    pd.DataFrame({"series": np.sin(np.linspace(0, 2 * np.pi, 32))}).to_csv(input_csv, index=False)
    # Εκτέλεση R script (υποθέτουμε ότι το script παράγει arima.csv)
    result = subprocess.run([
        "Rscript", str((Path(__file__).parents[1] / "r" / "time_series.R").resolve()),
        str(input_csv), str(output_csv)
    ], capture_output=True)
    assert result.returncode == 0, f"R error: {result.stderr.decode()}"
    # df = pd.read_csv(output_csv)
    # assert df.shape[1] >= 1

def test_neural_channel_coding():
    import sys
    sys.path.append(str((Path(__file__).parents[1] / "src" / "helios" / "information_theory").resolve()))
    import neural_channel_coding
    device = 'cuda' if neural_channel_coding.torch.cuda.is_available() else 'cpu'
    model = neural_channel_coding.train_autoencoder(device=device, epochs=150)
    ber = neural_channel_coding.eval_ber(model, device=device)
    assert ber < 0.15  # BER threshold for basic autoencoder


def test_wavelet_packet_julia():
    import subprocess
    import pandas as pd
    # Δημιουργία dummy σήματος και αποθήκευση σε CSV
    tmp_dir = Path("/tmp/helios_wavelet_packet")
    tmp_dir.mkdir(exist_ok=True)
    input_csv = tmp_dir / "signal.csv"
    output_csv = tmp_dir / "wpt.csv"
    pd.DataFrame({"signal": np.sin(np.linspace(0, 2 * np.pi, 32))}).to_csv(input_csv, index=False)
    # Εκτέλεση Julia script
    result = subprocess.run([
        "julia", str((Path(__file__).parents[1] / "julia" / "wavelet_packet.jl").resolve()),
        str(input_csv), str(output_csv), "db4", "2"
    ], capture_output=True)
    assert output_csv.exists(), f"Output CSV not created: {result.stderr.decode()}"
    df = pd.read_csv(output_csv)
    assert df.shape[1] >= 1


def test_octave_bode_bridge():
    import subprocess
    import pandas as pd
    tmp_dir = Path("/tmp/helios_octave_bode")
    tmp_dir.mkdir(exist_ok=True)
    output_csv = tmp_dir / "bode.csv"
    # Εκτέλεση Octave script (υποθέτουμε ότι το script παράγει bode.csv)
    result = subprocess.run([
        "octave", "--quiet", str((Path(__file__).parents[1] / "octave" / "filter_design.m").resolve())
    ], capture_output=True)
    # Δεν αποτυγχάνει η εκτέλεση
    assert result.returncode == 0, f"Octave error: {result.stderr.decode()}"
    # Έστω ότι το script σώζει bode.csv
    # df = pd.read_csv(output_csv)
    # assert df.shape[1] >= 2
