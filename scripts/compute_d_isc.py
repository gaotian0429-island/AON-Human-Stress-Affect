#!/usr/bin/env python3
"""Main panel D: 87-node high-minus-low ISC and participant summaries."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, ttest_1samp, wilcoxon

from isc_common import ROOT, contrast_weights, high_low, load_inputs, loo_isc


N_PERM = 20_000
SEED = 20260828


def center_runs(values, groups):
    result = np.asarray(values, float).copy()
    for run in np.unique(groups):
        selected = groups == run
        result[selected] -= result[selected].mean(axis=0, keepdims=True)
    return result


def circular_null_contrast(weights, values, groups, n_perm, seed):
    tables = []
    for run in np.unique(groups):
        selected = np.flatnonzero(groups == run)
        w, x = weights[selected], values[selected]
        tables.append(np.stack([np.roll(w, shift) @ x for shift in range(len(selected))]))
    rng = np.random.default_rng(seed)
    null = np.zeros((n_perm, values.shape[1]), float)
    for table in tables:
        null += table[rng.integers(0, len(table), size=n_perm)]
    return null


def circular_null_correlation(values, target, groups, n_perm, seed):
    tables = []
    for run in np.unique(groups):
        selected = np.flatnonzero(groups == run)
        x, y = values[selected], target[selected]
        tables.append(np.stack([x.T @ np.roll(y, shift) for shift in range(len(selected))]))
    rng = np.random.default_rng(seed)
    numerator = np.zeros((n_perm, values.shape[1]), float)
    for table in tables:
        numerator += table[rng.integers(0, len(table), size=n_perm)]
    denominator = np.sqrt((values * values).sum(0) * (target * target).sum())
    return numerator / (denominator[None, :] + 1e-30)


def ranking(subject_isc, names, runs, target, n_perm):
    group_isc = subject_isc.mean(axis=0).astype(float)
    low, high, _ = high_low(target, runs)
    weights = contrast_weights(target, runs)
    delta = weights @ group_isc
    delta_null = circular_null_contrast(weights, group_isc, runs, n_perm, SEED)
    delta_z = (delta - delta_null.mean(0)) / (delta_null.std(0, ddof=1) + 1e-30)
    delta_p = (1 + (delta_null >= delta[None, :]).sum(0)) / (n_perm + 1)

    centered_isc = center_runs(group_isc, runs)
    centered_target = center_runs(target[:, None], runs)[:, 0]
    continuous_r = (
        (centered_isc * centered_target[:, None]).sum(axis=0) /
        (np.sqrt((centered_isc * centered_isc).sum(axis=0) *
                 (centered_target * centered_target).sum()) + 1e-30)
    )
    continuous_null = circular_null_correlation(
        centered_isc, centered_target, runs, n_perm, SEED + 1
    )
    continuous_p = (1 + (continuous_null >= continuous_r[None, :]).sum(0)) / (n_perm + 1)
    continuous_z = ((continuous_r - continuous_null.mean(0)) /
                    (continuous_null.std(0, ddof=1) + 1e-30))

    frame = pd.DataFrame({
        "node": names,
        "normalize": True,
        "isc_low": group_isc[low].mean(0),
        "isc_high": group_isc[high].mean(0),
        "delta_isc": delta,
        "delta_permutation_z": delta_z,
        "delta_p_one_sided": delta_p,
        "continuous_isc_stress_related_affect_r": continuous_r,
        "continuous_permutation_z": continuous_z,
        "continuous_p_one_sided": continuous_p,
    })
    for rank_name, value_name in {
        "delta_raw_rank": "delta_isc",
        "delta_z_rank": "delta_permutation_z",
        "continuous_r_rank": "continuous_isc_stress_related_affect_r",
        "continuous_z_rank": "continuous_permutation_z",
    }.items():
        frame[rank_name] = rankdata(-frame[value_name], method="min").astype(int)
    n = len(frame)
    frame["human_two_metric_geomean"] = np.sqrt(
        ((n - frame["delta_z_rank"] + 1) / n) *
        ((n - frame["continuous_z_rank"] + 1) / n)
    )
    frame["human_raw_effect_geomean"] = np.sqrt(
        ((n - frame["delta_raw_rank"] + 1) / n) *
        ((n - frame["continuous_r_rank"] + 1) / n)
    )
    frame["human_raw_effect_joint_rank"] = rankdata(
        -frame["human_raw_effect_geomean"], method="min"
    ).astype(int)
    return frame


def subject_summary(subject_isc, names, runs, target):
    low, high, per_run = high_low(target, runs)
    pooled_delta = subject_isc[:, high].mean(1) - subject_isc[:, low].mean(1)
    run_delta = np.stack([
        subject_isc[:, run_high].mean(1) - subject_isc[:, run_low].mean(1)
        for run_low, run_high in per_run
    ], axis=1)
    equal_run_delta = run_delta.mean(1)
    rows = []
    for region, name in enumerate(names):
        delta = equal_run_delta[:, region]
        t = ttest_1samp(delta, 0, alternative="greater")
        try:
            w_p = float(wilcoxon(delta, alternative="greater", method="exact").pvalue)
        except ValueError:
            w_p = np.nan
        rows.append({
            "node": name,
            "isc_low_group_display": float(subject_isc[:, low, region].mean()),
            "isc_high_group_display": float(subject_isc[:, high, region].mean()),
            "isc_delta_group_display": float(pooled_delta[:, region].mean()),
            "subject_equal_run_delta_mean": float(delta.mean()),
            "subject_equal_run_delta_sd": float(delta.std(ddof=1)),
            "subject_positive_count": int(np.sum(delta > 0)),
            "subject_paired_t": float(t.statistic),
            "subject_paired_t_one_sided_nominal_p": float(t.pvalue),
            "subject_wilcoxon_one_sided_nominal_p": w_p,
            "subject_cohens_dz": float(delta.mean() / (delta.std(ddof=1) + 1e-30)),
            "subject_deltas": ";".join(f"{v:.8f}" for v in delta),
            "run_mean_deltas": ";".join(f"{v:.8f}" for v in run_delta[:, :, region].mean(0)),
        })
    frame = pd.DataFrame(rows)
    frame["paired_t_rank"] = rankdata(-frame["subject_paired_t"], method="min").astype(int)
    frame["delta_effect_rank"] = rankdata(
        -frame["subject_equal_run_delta_mean"], method="min"
    ).astype(int)
    return frame.sort_values("paired_t_rank")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=N_PERM)
    parser.add_argument("--output-root", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    if args.permutations < 1:
        parser.error("permutations must be positive")
    values, names, starts, runs, target = load_inputs()
    isc = loo_isc(values, starts)
    args.output_root.mkdir(parents=True, exist_ok=True)
    ranking(isc, names, runs, target, args.permutations).to_csv(
        args.output_root / "D_isc_ranking_87.csv", index=False
    )
    subject_summary(isc, names, runs, target).to_csv(
        args.output_root / "D_roi_isc_subject_summary.csv", index=False
    )
    np.savez_compressed(
        args.output_root / "isc_windows.npz", isc=isc, starts=starts, window_runs=runs
    )
    print(args.output_root)


if __name__ == "__main__":
    main()
