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


@dataclass
class NeuralChannelAutoencoder:
    input_bits: int = 4
    code_bits: int = 7
    hidden_dim: int = 16

    def theoretical_ber(self, snr_db: float) -> float:
        return float(turbo_code_ber(snr_db, rate=self.input_bits / self.code_bits) * 0.85)
