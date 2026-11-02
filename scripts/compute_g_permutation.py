#!/usr/bin/env python3
"""Strict train-only circular-shift permutation test for main panel G."""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import compute_fc as module
from isc_common import ROOT, ALIGNED, TIMESERIES, load_names


N_PERM = 1000
N_WORKERS = 16
SEED = 20260918

STATE = {}


def labels_from_target(target, groups):
    low = []
    high = []
    for run in np.unique(groups):
        positions = np.flatnonzero(groups == run)
        ordered = positions[np.argsort(target[positions], kind="mergesort")]
        count = len(positions) // 3
        low.extend(ordered[:count])
        high.extend(ordered[-count:])
    mask = np.zeros(len(target), bool)
    labels = np.full(len(target), -1, int)
    mask[low] = True
    mask[high] = True
    labels[low] = 0
    labels[high] = 1
    return mask, labels[mask]


def initialize(x, target, groups):
    STATE.update(x=x, target=target, groups=groups)


def one_permutation(index):
    rng = np.random.default_rng(SEED + 104729 * index)
    shifted = np.asarray(STATE["target"]).copy()
    groups = np.asarray(STATE["groups"])
    for run in np.unique(groups):
        positions = np.flatnonzero(groups == run)
        shifted[positions] = np.roll(shifted[positions], int(rng.integers(len(positions))))
    mask, labels = labels_from_target(shifted, groups)
    _, run_ba, run_auc, _ = module.nested_logistic_strict(
        np.asarray(STATE["x"])[mask], labels, groups[mask]
    )
    return float(run_ba.mean()), float(run_auc.mean())


def exact_signflip(values, chance=0.5):
    difference = np.asarray(values, float) - chance
    observed = float(difference.mean())
    null = [np.mean(difference * np.asarray(signs)) for signs in itertools.product((-1, 1), repeat=len(values))]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def run(label, target, out_dir):
    aligned = np.load(ALIGNED)
    starts = aligned["starts"].astype(int)
    groups = aligned["window_runs"].astype(int)
    values = np.load(TIMESERIES).astype(float)
    names = load_names()
    matrices = module.dynamic_fc_matrices(values.mean(axis=0), starts)
    aon = names.index("AON")
    x = matrices[:, aon, np.arange(len(names)) != aon].astype(float)
    mask, labels = labels_from_target(target, groups)
    _, run_ba, run_auc, c_values = module.nested_logistic_strict(x[mask], labels, groups[mask])
    observed = np.asarray([run_ba.mean(), run_auc.mean()])
    context = mp.get_context("fork")
    with context.Pool(N_WORKERS, initializer=initialize, initargs=(x, target, groups)) as pool:
        null = np.asarray(list(pool.imap(one_permutation, range(N_PERM), chunksize=1)))
    p = (1 + np.sum(null >= observed[None, :], axis=0)) / (N_PERM + 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "node": "AON",
        "label_lag_seconds": 6,
        "balanced_accuracy": observed[0],
        "auc": observed[1],
        "positive_ba_runs": int(np.sum(run_ba > 0.5)),
        "ba_run_signflip_one_sided_p": exact_signflip(run_ba),
        "auc_run_signflip_one_sided_p": exact_signflip(run_auc),
        "ba_circular_full_refit_one_sided_p": float(p[0]),
        "auc_circular_full_refit_one_sided_p": float(p[1]),
        "run_balanced_accuracy": ";".join(f"{v:.6f}" for v in run_ba),
        "run_auc": ";".join(f"{v:.6f}" for v in run_auc),
        "selected_c": ";".join(f"{v:g}" for v in c_values),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "G_aon_classifier_permutation_summary.csv", index=False)
    pd.DataFrame({
        "permutation_index": np.arange(N_PERM),
        "balanced_accuracy": null[:, 0],
        "roc_auc": null[:, 1],
        "observed_balanced_accuracy": observed[0],
        "observed_roc_auc": observed[1],
    }).to_csv(out_dir / "G_aon_classifier_permutation_null.csv", index=False)
    (out_dir / "g_permutation_metadata.json").write_text(json.dumps({
        "method": "strict train-only nested leave-one-run-out circular-shift permutation",
        "label": label,
        "n_permutations": N_PERM,
        "seed": SEED,
        "workers": N_WORKERS,
        "window_seconds": 30,
        "step_seconds": 5,
        "label_lag_seconds": 6,
        "state_definition": "top/bottom third separately within each run",
        "centering": "training-run means estimated inside every outer and inner fold; held-out run uses equal-weight training-run means",
    }, indent=2) + "\n")
    print(label, json.dumps({"observed": observed.tolist(), "p": p.tolist()}, indent=2), flush=True)


def main():
    global N_PERM, N_WORKERS
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=N_PERM)
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--output-root", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    N_PERM, N_WORKERS = args.permutations, args.workers
    if N_PERM < 1 or N_WORKERS < 1:
        parser.error("permutations and workers must be positive")
    aligned = np.load(ALIGNED)
    run("main_stress_related_affect", aligned["anxiety_windows"].astype(float), args.output_root)


if __name__ == "__main__":
    main()
