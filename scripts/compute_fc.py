from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from isc_common import ROOT, ALIGNED, TIMESERIES, load_names


STATE = {}


def correlation(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.sqrt(a @ a * (b @ b)) + 1e-30))


def dynamic_fc_matrices(values, starts, width=30):
    output = np.empty((len(starts), values.shape[1], values.shape[1]), np.float32)
    for index, start in enumerate(starts):
        window = values[start:start + width].astype(float)
        window -= window.mean(axis=0, keepdims=True)
        covariance = window.T @ window
        scale = np.sqrt(np.diag(covariance))
        r = covariance / (scale[:, None] * scale[None, :] + 1e-30)
        output[index] = np.arctanh(np.clip(r, -0.999999, 0.999999))
    return output


def fit_run_center(x, groups, train_mask):
    means = {}
    for run in np.unique(groups[train_mask]):
        means[int(run)] = x[train_mask & (groups == run)].mean(axis=0)
    pooled = np.mean(np.stack(list(means.values())), axis=0)
    return means, pooled


def apply_run_center(x, groups, stats):
    means, pooled = stats
    out = np.asarray(x, float).copy()
    for run in np.unique(groups):
        out[groups == run] -= means.get(int(run), pooled)
    return out


def fit_logistic(x_train, y_train, x_test, c_value):
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1
    train = (x_train - mean) / scale
    test = (x_test - mean) / scale
    model = LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="liblinear",
        max_iter=2000,
        class_weight="balanced",
        random_state=20260828,
    )
    model.fit(train, y_train)
    return model.predict_proba(test)[:, 1]


def nested_logistic_strict(x, labels, groups):
    runs = np.unique(groups)
    c_grid = np.logspace(-3, 3, 7)
    probability = np.full(len(labels), np.nan)
    selected_c = []
    for test_run in runs:
        outer_train = groups != test_run
        outer_test = ~outer_train
        scores = []
        for c_value in c_grid:
            inner_scores = []
            for validation_run in runs[runs != test_run]:
                inner_train = outer_train & (groups != validation_run)
                inner_valid = groups == validation_run
                stats = fit_run_center(x, groups, inner_train)
                x_train = apply_run_center(x[inner_train], groups[inner_train], stats)
                x_valid = apply_run_center(x[inner_valid], groups[inner_valid], stats)
                p = fit_logistic(x_train, labels[inner_train], x_valid, c_value)
                inner_scores.append(
                    balanced_accuracy_score(labels[inner_valid], p >= 0.5)
                )
            scores.append(np.mean(inner_scores))
        c_value = float(c_grid[int(np.argmax(scores))])
        stats = fit_run_center(x, groups, outer_train)
        x_train = apply_run_center(x[outer_train], groups[outer_train], stats)
        x_test = apply_run_center(x[outer_test], groups[outer_test], stats)
        probability[outer_test] = fit_logistic(
            x_train, labels[outer_train], x_test, c_value
        )
        selected_c.append(c_value)
    run_ba = []
    run_auc = []
    for run in runs:
        selected = groups == run
        run_ba.append(balanced_accuracy_score(labels[selected], probability[selected] >= 0.5))
        run_auc.append(roc_auc_score(labels[selected], probability[selected]))
    return probability, np.asarray(run_ba), np.asarray(run_auc), selected_c


def fit_target_center(y, groups, train_mask):
    means = {}
    for run in np.unique(groups[train_mask]):
        means[int(run)] = float(y[train_mask & (groups == run)].mean())
    pooled = float(np.mean(list(means.values())))
    return means, pooled


def apply_target_center(y, groups, stats):
    means, pooled = stats
    out = np.asarray(y, float).copy()
    for run in np.unique(groups):
        out[groups == run] -= means.get(int(run), pooled)
    return out


def fit_ridge(x_train, y_train, x_test, alpha):
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1
    model = Ridge(alpha=float(alpha), fit_intercept=False, solver="cholesky")
    model.fit(x_train / scale, y_train)
    return model.predict(x_test / scale)


def nested_ridge_strict(x, y, groups):
    runs = np.unique(groups)
    alphas = np.logspace(-2, 5, 15)
    prediction = np.zeros(len(y), float)
    test_centered_target = np.zeros(len(y), float)
    selected_alphas = []
    for test_run in runs:
        outer_train = groups != test_run
        outer_test = ~outer_train
        scores = []
        for alpha in alphas:
            inner_scores = []
            for validation_run in runs[runs != test_run]:
                inner_train = outer_train & (groups != validation_run)
                inner_valid = groups == validation_run
                x_stats = fit_run_center(x, groups, inner_train)
                y_stats = fit_target_center(y, groups, inner_train)
                x_train = apply_run_center(x[inner_train], groups[inner_train], x_stats)
                x_valid = apply_run_center(x[inner_valid], groups[inner_valid], x_stats)
                y_train = apply_target_center(y[inner_train], groups[inner_train], y_stats)
                p = fit_ridge(x_train, y_train, x_valid, alpha)
                inner_scores.append(correlation(p, y[inner_valid]))
            scores.append(np.mean(inner_scores))
        alpha = float(alphas[int(np.argmax(scores))])
        x_stats = fit_run_center(x, groups, outer_train)
        y_stats = fit_target_center(y, groups, outer_train)
        x_train = apply_run_center(x[outer_train], groups[outer_train], x_stats)
        x_test = apply_run_center(x[outer_test], groups[outer_test], x_stats)
        y_train = apply_target_center(y[outer_train], groups[outer_train], y_stats)
        prediction[outer_test] = fit_ridge(x_train, y_train, x_test, alpha)
        test_centered_target[outer_test] = apply_target_center(
            y[outer_test], groups[outer_test], y_stats
        )
        selected_alphas.append(alpha)
    run_r = np.asarray([
        correlation(prediction[groups == run], y[groups == run]) for run in runs
    ])
    return prediction, test_centered_target, run_r, selected_alphas


def initialize_worker(matrices, target, groups, labels, class_mask, names, aon_index):
    STATE.update(
        matrices=matrices,
        target=target,
        groups=groups,
        labels=labels,
        class_mask=class_mask,
        names=names,
        aon_index=aon_index,
    )


def evaluate_seed(seed):
    matrices = STATE["matrices"]
    target = STATE["target"]
    groups = STATE["groups"]
    names = STATE["names"]
    keep = np.arange(matrices.shape[1]) != seed
    x = matrices[:, seed, keep].astype(float)
    prediction, centered_target, run_r, alphas = nested_ridge_strict(x, target, groups)
    class_mask = STATE["class_mask"]
    labels = STATE["labels"]
    probability, run_ba, run_auc, c_values = nested_logistic_strict(
        x[class_mask], labels, groups[class_mask]
    )
    row = {
        "node": names[seed],
        "n_fc_features": int(x.shape[1]),
        "regression_mean_heldout_run_r": float(run_r.mean()),
        "regression_fisher_mean_heldout_run_r": float(
            np.tanh(np.mean(np.arctanh(np.clip(run_r, -0.999999, 0.999999))))
        ),
        "regression_pooled_r": float(correlation(prediction, target)),
        "regression_positive_runs": int(np.sum(run_r > 0)),
        "regression_run_r": ";".join(f"{v:.6f}" for v in run_r),
        "regression_alpha": ";".join(f"{v:g}" for v in alphas),
        "classification_mean_balanced_accuracy": float(run_ba.mean()),
        "classification_pooled_balanced_accuracy": float(
            balanced_accuracy_score(labels, probability >= 0.5)
        ),
        "classification_mean_auc": float(run_auc.mean()),
        "classification_pooled_auc": float(roc_auc_score(labels, probability)),
        "classification_runs_above_chance": int(np.sum(run_ba > 0.5)),
        "classification_run_balanced_accuracy": ";".join(f"{v:.6f}" for v in run_ba),
        "classification_run_auc": ";".join(f"{v:.6f}" for v in run_auc),
        "classification_c": ";".join(f"{v:g}" for v in c_values),
    }
    detail = None
    if seed == STATE["aon_index"]:
        detail = {
            "prediction": prediction,
            "centered_target": centered_target,
            "run_r": run_r,
            "alphas": alphas,
        }
    return seed, row, detail


def run_one(label, target, output_root, workers):
    aligned = np.load(ALIGNED)
    starts = aligned["starts"].astype(int)
    groups = aligned["window_runs"].astype(int)
    values = np.load(TIMESERIES).astype(float)
    names = load_names()
    if values.shape != (10, 6073, 87) or len(names) != 87 or len(starts) != 1169:
        raise ValueError("Expected 10 subjects, 6073 TRs, 87 nodes and 1169 windows")
    matrices = dynamic_fc_matrices(values.mean(axis=0), starts)
    low = []
    high = []
    for run in np.unique(groups):
        positions = np.flatnonzero(groups == run)
        ordered = positions[np.argsort(target[positions], kind="mergesort")]
        count = len(positions) // 3
        low.extend(ordered[:count])
        high.extend(ordered[-count:])
    class_mask = np.zeros(len(groups), bool)
    labels_full = np.full(len(groups), -1, int)
    class_mask[low] = True
    class_mask[high] = True
    labels_full[low] = 0
    labels_full[high] = 1
    labels = labels_full[class_mask]
    aon_index = names.index("AON")
    context = mp.get_context("fork")
    with context.Pool(
        workers,
        initializer=initialize_worker,
        initargs=(matrices, target, groups, labels, class_mask, names, aon_index),
    ) as pool:
        results = list(pool.imap(evaluate_seed, range(len(names)), chunksize=1))
    results.sort(key=lambda x: x[0])
    frame = pd.DataFrame([x[1] for x in results])
    frame["regression_rank"] = frame["regression_mean_heldout_run_r"].rank(
        ascending=False, method="min"
    ).astype(int)
    frame["classification_balanced_accuracy_rank"] = frame[
        "classification_mean_balanced_accuracy"
    ].rank(ascending=False, method="min").astype(int)
    frame["classification_auc_rank"] = frame["classification_mean_auc"].rank(
        ascending=False, method="min"
    ).astype(int)
    out = output_root / label
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "G_roi_fc_observed_87.csv", index=False)
    detail = next(x[2] for x in results if x[2] is not None)
    raw_name = "stress_related_affect_30s_mean"
    centered_name = "observed_stress_related_affect_train_only_centered"
    pred_name = "nested_loro_predicted_stress_related_affect"
    pd.DataFrame({
        "window_index": np.arange(len(groups)),
        "run": groups,
        "fmri_window_start_s": starts,
        raw_name: target,
        centered_name: detail["centered_target"],
        pred_name: detail["prediction"],
    }).to_csv(out / "H_nested_loro_predictions.csv", index=False)
    pd.DataFrame({
        "run": np.unique(groups),
        "heldout_run_r": detail["run_r"],
        "selected_ridge_alpha": detail["alphas"],
    }).to_csv(out / "H_nested_loro_run_statistics.csv", index=False)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "output")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "method": "strict train-only run centering inside every outer and inner CV fold",
        "feature_policy": "training-run means estimated from training runs; held-out run uses the equal-weight mean of training-run means",
        "target_policy": "training-run means estimated from training runs; held-out target is not used to fit preprocessing",
        "input": "repository data directory containing frozen 87-node time series and aligned annotations",
        "note": "Pearson run correlations are evaluated against the raw held-out target; output centered target is fold-specific and diagnostic.",
    }
    (args.output_root / "fc_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    main_out = run_one(
        "main_stress_related_affect",
        np.load(ALIGNED)["anxiety_windows"].astype(float),
        args.output_root,
        args.workers,
    )
    print(main_out)


if __name__ == "__main__":
    main()
