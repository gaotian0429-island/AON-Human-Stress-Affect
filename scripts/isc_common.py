"""Shared frozen-input and leave-one-participant-out ISC calculations."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "input"
TIMESERIES = INPUT / "unique87_timeseries.npy"
NAMES = ROOT / "roi_87_nodes.csv"
ALIGNED = INPUT / "aligned_annotations_30s5s.npz"
EMOTIONS = INPUT / "fall_emo_annot_zscore_mean_cut.csv"
WIDTH = 30
LAG = 6


def load_names():
    with NAMES.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 87 or any(int(row["node_index"]) != i for i, row in enumerate(rows, 1)):
        raise ValueError("ROI metadata must contain 87 nodes in array-column order")
    return [row["roi_abbreviation"] for row in rows]


def load_inputs():
    values = np.load(TIMESERIES).astype(float)
    names = load_names()
    with np.load(ALIGNED) as aligned:
        starts = aligned["starts"].astype(int)
        runs = aligned["window_runs"].astype(int)
        target = aligned["anxiety_windows"].astype(float)
        anxiety = aligned["anxiety_only_windows"].astype(float)
        fear = aligned["fearful_windows"].astype(float)
    if values.shape != (10, 6073, 87) or len(names) != 87:
        raise ValueError("Expected 10 participants x 6073 TRs x 87 nodes")
    if len(starts) != 1169 or len(runs) != 1169 or len(target) != 1169:
        raise ValueError("Expected 1169 aligned windows")
    if not np.allclose(target, (anxiety + fear) / 2, atol=1e-12):
        raise ValueError("Stress-related affect must equal mean(z-anxiety, z-fear)")
    if not np.array_equal(np.unique(runs), np.arange(1, 8)):
        raise ValueError("Expected seven runs numbered 1 to 7")
    if np.any(starts < 0) or np.any(starts + WIDTH > values.shape[1]):
        raise ValueError("Windows extend beyond the frozen time series")
    return values, names, starts, runs, target


def loo_isc(values, starts):
    """Subject x window x node Pearson ISC, matching the frozen analysis."""
    indices = starts[:, None] + np.arange(WIDTH)[None, :]
    n_subjects, _, n_regions = values.shape
    result = np.empty((n_subjects, len(starts), n_regions), np.float32)
    for lo in range(0, n_regions, 16):
        hi = min(lo + 16, n_regions)
        windowed = values[:, indices, lo:hi].astype(float)
        windowed -= windowed.mean(axis=2, keepdims=True)
        other = (windowed.sum(axis=0, keepdims=True) - windowed) / (n_subjects - 1)
        numerator = np.sum(windowed * other, axis=2)
        denominator = np.sqrt(
            np.sum(windowed * windowed, axis=2) *
            np.sum(other * other, axis=2)
        )
        result[:, :, lo:hi] = numerator / (denominator + 1e-12)
    return result


def high_low(target, runs):
    high, low, per_run = [], [], []
    for run in np.unique(runs):
        positions = np.flatnonzero(runs == run)
        order = positions[np.argsort(target[positions], kind="mergesort")]
        n = len(positions) // 3
        run_low, run_high = order[:n], order[-n:]
        low.extend(run_low)
        high.extend(run_high)
        per_run.append((run_low, run_high))
    return np.asarray(low), np.asarray(high), per_run


def contrast_weights(target, runs):
    low, high, _ = high_low(target, runs)
    weights = np.zeros(len(target), float)
    weights[high] = 1 / len(high)
    weights[low] = -1 / len(low)
    return weights
