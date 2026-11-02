#!/usr/bin/env python3
"""Main panel F: paired AON ISC contrasts against five other emotions."""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from isc_common import EMOTIONS, LAG, ROOT, WIDTH, contrast_weights, load_inputs, loo_isc


EMOTION_COLUMNS = {
    "Angry": "angry",
    "Disgusted": "disgusted",
    "Happy": "happy",
    "Sad": "sad",
    "Surprise": "surprised",
}


def exact_sign_flip(values, alternative):
    x = np.asarray(values, float)
    signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=len(x))))
    null = signs * x[None, :]
    means = null.mean(axis=1)
    sds = null.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        null_t = means / (sds / np.sqrt(len(x)))
    observed_t = stats.ttest_1samp(x, 0).statistic
    if alternative == "greater":
        p = np.mean(null_t >= observed_t - 1e-12)
    elif alternative == "two-sided":
        p = np.mean(np.abs(null_t) >= abs(observed_t) - 1e-12)
    else:
        raise ValueError(alternative)
    return float(observed_t), float(p)


def bh_fdr(pvalues):
    pvalues = np.asarray(pvalues, float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def summarize(values):
    x = np.asarray(values, float)
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / np.sqrt(len(x))
    critical = stats.t.ppf(0.975, len(x) - 1)
    statistic, p_one = exact_sign_flip(x, "greater")
    _, p_two = exact_sign_flip(x, "two-sided")
    return {
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95_low": mean - critical * se,
        "ci95_high": mean + critical * se,
        "t": statistic,
        "exact_one_sided_p": p_one,
        "exact_two_sided_p": p_two,
        "positive_n": int(np.sum(x > 0)),
        "cohens_dz": mean / sd if sd > 0 else np.nan,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    values, names, starts, runs, target = load_inputs()
    frame = pd.read_csv(EMOTIONS)
    if len(frame) != 6073 or not set(EMOTION_COLUMNS.values()).issubset(frame.columns):
        raise ValueError("Emotion CSV must contain 6073 TRs and five emotion columns")
    cache = args.output_root / "isc_windows.npz"
    if cache.exists():
        with np.load(cache) as loaded:
            isc = loaded["isc"]
            if not np.array_equal(loaded["starts"], starts) or not np.array_equal(loaded["window_runs"], runs):
                raise ValueError("ISC cache does not match aligned windows")
    else:
        isc = loo_isc(values, starts)
    if isc.shape != (10, 1169, 87):
        raise ValueError("Expected 10 x 1169 x 87 ISC array")
    aon = names.index("AON")
    annotation_indices = starts[:, None] + np.arange(WIDTH)[None, :] - LAG
    if np.any(annotation_indices < 0) or np.any(annotation_indices >= len(frame)):
        raise ValueError("Emotion indices extend beyond the aligned annotation CSV")
    annotations = [target]
    for column in EMOTION_COLUMNS.values():
        annotations.append(frame[column].to_numpy(float)[annotation_indices].mean(axis=1))
    annotations = np.stack(annotations)
    weights = np.stack([contrast_weights(x, runs) for x in annotations])
    fisher_isc = np.arctanh(np.clip(isc.astype(np.float64), -0.999999, 0.999999))
    subject_effect_z = np.einsum("ew,swr->ser", weights, fisher_isc, optimize=True)
    composite = subject_effect_z[:, 0, aon]
    rows = []
    for emotion_index, emotion in enumerate(EMOTION_COLUMNS, start=1):
        difference = composite - subject_effect_z[:, emotion_index, aon]
        rows.append({
            "comparison": f"Stress-related affect - {emotion}",
            "emotion": emotion,
            **summarize(difference),
        })
    comparisons = pd.DataFrame(rows)
    comparisons["BH_FDR_q_five_one_sided"] = bh_fdr(comparisons["exact_one_sided_p"])
    comparisons["BH_FDR_q_five_two_sided"] = bh_fdr(comparisons["exact_two_sided_p"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(args.output_root / "F_stress_related_vs_five_emotions.csv", index=False)
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
