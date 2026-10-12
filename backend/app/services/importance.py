"""Deterministic statistical and importance calculation algorithms."""
import math
from typing import Dict, List, Optional, Tuple


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def variance(values: List[float], sample_mean: Optional[float] = None) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = sample_mean if sample_mean is not None else mean(values)
    return sum((x - m) ** 2 for x in values) / (n - 1)


def std_dev(values: List[float], sample_mean: Optional[float] = None) -> float:
    var = variance(values, sample_mean)
    return math.sqrt(var) if var > 0.0 else 0.0


def pearson_correlation(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0
    mx = mean(x)
    my = mean(y)
    var_x = sum((xi - mx) ** 2 for xi in x)
    var_y = sum((yi - my) ** 2 for yi in y)
    if var_x <= 1e-12 or var_y <= 1e-12:
        return 0.0
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denom = math.sqrt(var_x * var_y)
    if denom <= 1e-12:
        return 0.0
    r = cov / denom
    # Clamp to [-1.0, 1.0] to prevent floating point inaccuracies
    return max(-1.0, min(1.0, r))


def rank_data(values: List[float]) -> List[float]:
    n = len(values)
    indices = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and abs(values[indices[j]] - values[indices[j + 1]]) < 1e-12:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indices[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_correlation(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    rx = rank_data(x)
    ry = rank_data(y)
    return pearson_correlation(rx, ry)


def compute_quantiles(values: List[float], num_bins: int = 5) -> List[float]:
    if not values:
        return []
    s = sorted(values)
    n = len(s)
    thresholds = []
    for b in range(1, num_bins):
        idx = int(b * n / num_bins)
        thresholds.append(s[min(idx, n - 1)])
    return sorted(list(set(thresholds)))


def encode_quantile_bin(val: float, thresholds: List[float]) -> int:
    for i, t in enumerate(thresholds):
        if val <= t:
            return i
    return len(thresholds)


def cyclical_sin(val: float, period: float) -> float:
    if period <= 0:
        return 0.0
    return math.sin(2.0 * math.pi * val / period)


def cyclical_cos(val: float, period: float) -> float:
    if period <= 0:
        return 1.0
    return math.cos(2.0 * math.pi * val / period)
