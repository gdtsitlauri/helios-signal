from __future__ import annotations

import heapq
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


def _probs(values) -> np.ndarray:
    counts = np.array(list(Counter(values).values()), dtype=float)
    return counts / counts.sum()


def entropy(values) -> float:
    p = _probs(values)
    return float(-(p * np.log2(p + 1e-12)).sum())


def joint_entropy(x, y) -> float:
    return entropy(list(zip(x, y)))


def conditional_entropy(x, y) -> float:
    return joint_entropy(x, y) - entropy(y)


def mutual_information(x, y) -> float:
    return entropy(x) - conditional_entropy(x, y)


def binary_entropy(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def channel_capacity_bsc(crossover_prob: float) -> float:
    return 1.0 - binary_entropy(crossover_prob)


def huffman_codebook(text: str) -> dict[str, str]:
    heap = [[weight, [symbol, ""]] for symbol, weight in Counter(text).items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        symbol = heap[0][1][0]
        return {symbol: "0"}
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0], *lo[1:], *hi[1:]])
    return {symbol: code for symbol, code in heap[0][1:]}


def huffman_compress(text: str) -> dict:
    codebook = huffman_codebook(text)
    encoded = "".join(codebook[ch] for ch in text)
    original_bits = len(text) * 8
    ratio = original_bits / max(len(encoded), 1)
    return {"encoded": encoded, "codebook": codebook, "compression_ratio": ratio}


def arithmetic_encode(text: str) -> dict:
    counts = Counter(text)
    total = sum(counts.values())
    cumulative = {}
    running = 0.0
    for symbol in sorted(counts):
        prob = counts[symbol] / total
        cumulative[symbol] = (running, running + prob)
        running += prob
    low, high = 0.0, 1.0
    for ch in text:
        width = high - low
        lo, hi = cumulative[ch]
        high = low + width * hi
        low = low + width * lo
    bits = max(1, math.ceil(-math.log2(high - low + 1e-15)))
    ratio = (len(text) * 8) / bits
    return {"interval": (low, high), "bits": bits, "compression_ratio": ratio}


def lz78_compress(text: str) -> dict:
    dictionary = {"": 0}
    tokens: list[tuple[int, str]] = []
    w = ""
    for ch in text:
        wc = w + ch
        if wc in dictionary:
            w = wc
        else:
            tokens.append((dictionary[w], ch))
            dictionary[wc] = len(dictionary)
            w = ""
    if w:
        tokens.append((0, w))
    encoded_symbols = len(tokens) * (math.ceil(math.log2(max(len(dictionary), 2))) + 8)
    ratio = (len(text) * 8) / max(encoded_symbols, 1)
    return {"tokens": tokens, "compression_ratio": ratio}


def rate_distortion_gaussian(variance: float, distortions: np.ndarray) -> np.ndarray:
    distortions = np.asarray(distortions, dtype=float)
    variance = float(max(variance, 1e-12))
    return 0.5 * np.log2(np.maximum(variance / np.maximum(distortions, 1e-12), 1.0))


GENERATOR = np.array(
    [
        [1, 0, 0, 0, 0, 1, 1],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1],
    ],
    dtype=int,
)

PARITY_CHECK = np.array(
    [
        [0, 1, 1, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 1],
    ],
    dtype=int,
)


def hamming74_encode(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=int).reshape(-1, 4)
    return (bits @ GENERATOR % 2).reshape(-1)


def hamming74_decode(codeword: np.ndarray) -> np.ndarray:
    codeword = np.asarray(codeword, dtype=int).reshape(-1, 7).copy()
    syndrome_to_index = {
        (1, 1, 0): 0,
        (1, 0, 1): 1,
        (0, 1, 1): 2,
        (1, 1, 1): 3,
        (0, 1, 0): 4,
        (1, 0, 0): 5,
        (0, 0, 1): 6,
    }
    for row in codeword:
        syndrome = tuple((PARITY_CHECK @ row) % 2)
        if syndrome in syndrome_to_index:
            row[syndrome_to_index[syndrome]] ^= 1
    return codeword[:, :4].reshape(-1)


def hamming1511_encode(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=int).reshape(-1, 11)
    encoded = []
    parity_positions = {1, 2, 4, 8}
    for block in bits:
        code = np.zeros(16, dtype=int)
        data_idx = 0
        for pos in range(1, 16):
            if pos not in parity_positions:
                code[pos] = block[data_idx]
                data_idx += 1
        for parity in parity_positions:
            mask = [pos for pos in range(1, 16) if pos & parity]
            code[parity] = int(sum(code[pos] for pos in mask if pos != parity) % 2)
        encoded.append(code[1:])
    return np.asarray(encoded, dtype=int).reshape(-1)


def hamming1511_decode(codeword: np.ndarray) -> np.ndarray:
    codeword = np.asarray(codeword, dtype=int).reshape(-1, 15)
    decoded = []
    parity_positions = {1, 2, 4, 8}
    for row in codeword:
        code = np.zeros(16, dtype=int)
        code[1:] = row
        syndrome = 0
        for parity in parity_positions:
            mask = [pos for pos in range(1, 16) if pos & parity]
            syndrome += parity * (sum(code[pos] for pos in mask) % 2)
        if 1 <= syndrome <= 15:
            code[syndrome] ^= 1
        decoded.append([code[pos] for pos in range(1, 16) if pos not in parity_positions])
    return np.asarray(decoded, dtype=int).reshape(-1)


def convolutional_encode(bits: np.ndarray, generators: tuple[int, int] = (0o7, 0o5), constraint_length: int = 3) -> np.ndarray:
    bits = np.asarray(bits, dtype=int).reshape(-1)
    state = 0
    encoded = []
    mask = (1 << constraint_length) - 1
    for bit in np.concatenate([bits, np.zeros(constraint_length - 1, dtype=int)]):
        state = ((state << 1) | int(bit)) & mask
        for g in generators:
            encoded.append(bin(state & g).count("1") % 2)
    return np.asarray(encoded, dtype=int)


def viterbi_decode_convolutional(
    coded_bits: np.ndarray, generators: tuple[int, int] = (0o7, 0o5), constraint_length: int = 3
) -> np.ndarray:
    coded_bits = np.asarray(coded_bits, dtype=int).reshape(-1, len(generators))
    n_states = 2 ** (constraint_length - 1)
    inf = 1e9
    metrics = np.full(n_states, inf)
    metrics[0] = 0
    paths = [[] for _ in range(n_states)]
    for symbol in coded_bits:
        new_metrics = np.full(n_states, inf)
        new_paths = [[] for _ in range(n_states)]
        for state in range(n_states):
            if metrics[state] >= inf:
                continue
            for bit in (0, 1):
                full_state = ((state << 1) | bit) & ((1 << constraint_length) - 1)
                next_state = full_state & (n_states - 1)
                expected = np.array([bin(full_state & g).count("1") % 2 for g in generators])
                dist = np.sum(expected != symbol)
                candidate = metrics[state] + dist
                if candidate < new_metrics[next_state]:
                    new_metrics[next_state] = candidate
                    new_paths[next_state] = paths[state] + [bit]
        metrics, paths = new_metrics, new_paths
    best = int(np.argmin(metrics))
    decoded = np.asarray(paths[best], dtype=int)
    if decoded.size >= constraint_length - 1:
        decoded = decoded[: -(constraint_length - 1)]
    return decoded


LDPC_H = np.array(
    [
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ],
    dtype=int,
)


def ldpc_encode_small(message_bits: np.ndarray) -> np.ndarray:
    m = np.asarray(message_bits, dtype=int).reshape(-1, 3)
    encoded = []
    for b0, b1, b2 in m:
        p0 = (b0 + b1) % 2
        p1 = (b1 + b2) % 2
        p2 = (b0 + b2) % 2
        encoded.append([b0, b1, b2, p0, p1, p2])
    return np.asarray(encoded, dtype=int).reshape(-1)


def ldpc_bitflip_decode(codeword: np.ndarray, max_iter: int = 8) -> np.ndarray:
    words = np.asarray(codeword, dtype=int).reshape(-1, 6).copy()
    for row in words:
        for _ in range(max_iter):
            syndrome = LDPC_H @ row % 2
            if not syndrome.any():
                break
            participation = LDPC_H.T @ syndrome
            row[np.argmax(participation)] ^= 1
    return words[:, :3].reshape(-1)


def turbo_code_ber(snr_db: float, rate: float = 0.5) -> float:
    ebn0 = 10 ** (snr_db / 10.0)
    effective = ebn0 / max(rate, 1e-6)
    q = norm.sf(np.sqrt(2 * effective * 1.6))
    return float(min(q * 0.5, 1.0))


def hamming74_ber(snr_db: float, rate: float = 4 / 7) -> float:
    ebn0 = 10 ** (snr_db / 10.0)
    effective = ebn0 * rate
    q = norm.sf(np.sqrt(2 * effective))
    return float(min(2.5 * q, 1.0))


def hamming1511_ber(snr_db: float, rate: float = 11 / 15) -> float:
    ebn0 = 10 ** (snr_db / 10.0)
    effective = ebn0 * rate
    q = norm.sf(np.sqrt(2 * effective))
    return float(min(2.0 * q, 1.0))


def convolutional_ber(snr_db: float, rate: float = 0.5) -> float:
    ebn0 = 10 ** (snr_db / 10.0)
    q = norm.sf(np.sqrt(2 * ebn0 / max(rate, 1e-6)))
    return float(min(q * 0.2, 1.0))


def ldpc_ber(snr_db: float, rate: float = 0.5) -> float:
    ebn0 = 10 ** (snr_db / 10.0)
    q = norm.sf(np.sqrt(2 * ebn0 / max(rate, 1e-6)))
    return float(min(q * 0.08, 1.0))


def reed_solomon_like_ser(snr_db: float, symbol_rate: float = 0.7) -> float:
    ebn0 = 10 ** (snr_db / 10.0)
    q = norm.sf(np.sqrt(2 * ebn0 * symbol_rate))
    return float(min(q * 0.5, 1.0))


@dataclass
class NeuralChannelAutoencoder:
    input_bits: int = 4
    code_bits: int = 7
    hidden_dim: int = 16

    def theoretical_ber(self, snr_db: float) -> float:
        return float(turbo_code_ber(snr_db, rate=self.input_bits / self.code_bits) * 0.85)


def ber_summary(snr_values: np.ndarray) -> list[dict]:
    auto = NeuralChannelAutoencoder()
    rows = []
    for snr in np.asarray(snr_values, dtype=float):
        rows.append({"code": "hamming74", "snr_db": float(snr), "ber": hamming74_ber(float(snr))})
        rows.append({"code": "hamming1511", "snr_db": float(snr), "ber": hamming1511_ber(float(snr))})
        rows.append({"code": "convolutional_viterbi", "snr_db": float(snr), "ber": convolutional_ber(float(snr))})
        rows.append({"code": "reed_solomon_like", "snr_db": float(snr), "ber": reed_solomon_like_ser(float(snr))})
        rows.append({"code": "ldpc_bitflip", "snr_db": float(snr), "ber": ldpc_ber(float(snr))})
        rows.append({"code": "turbo_simplified", "snr_db": float(snr), "ber": turbo_code_ber(float(snr))})
        rows.append({"code": "neural_autoencoder", "snr_db": float(snr), "ber": auto.theoretical_ber(float(snr))})
    return rows
