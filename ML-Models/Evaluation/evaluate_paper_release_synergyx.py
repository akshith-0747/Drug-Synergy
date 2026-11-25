"""Evaluate the project's cross-modal attention model on author-release features.

This is an internal research comparison. It does not claim an exact recreation
of the paper's SynergyX implementation.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)

from ML.evaluation.paper_release_feature_benchmark import (
    ML_DIR,
    RESULTS_DIR,
    _build_candidate_rows,
)
from ML.feature_engineering.features import make_feature_matrix
from ML.models.synergyx import SynergyXRegressor
from ML.storage import load
from ML.training.train import _metrics, _stratification_labels


def _inner_split(train_idx, y, groups, protocol, seed):
    if protocol == "pair_grouped":
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        inner_train, inner_valid = next(
            splitter.split(np.zeros(len(train_idx)), y[train_idx], groups[train_idx])
        )
    else:
        labels = _stratification_labels(y[train_idx], folds=5)
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        inner_train, inner_valid = next(splitter.split(np.zeros(len(train_idx)), labels))
    return train_idx[inner_train], train_idx[inner_valid]


def run(protocol: str, repeats: int, epochs: int, threads: int, base_seed: int) -> dict:
    started = time.time()
    torch.set_num_threads(threads)
    frame, X, y, groups, coverage = _build_candidate_rows()
    context = load(ML_DIR / "results/models/feature_context.joblib")
    _, _, drs_rows, drs_counts = make_feature_matrix(frame.copy(), "drs", context)
    drs_row_ids = set(drs_rows["_benchmark_row_id"].astype(np.int64).tolist())
    drs_presence = frame["_benchmark_row_id"].astype(np.int64).isin(drs_row_ids).to_numpy()
    labels = _stratification_labels(y, folds=5)
    cell_dim = int(coverage["cell_feature_dim"])
    drug_dim = int((X.shape[1] - cell_dim) // 2)
    results = []

    for repeat in range(repeats):
        seed = base_seed + repeat
        if protocol == "pair_grouped":
            splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            splits = splitter.split(X, labels, groups)
        else:
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            splits = splitter.split(X, labels)

        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            inner_train, inner_valid = _inner_split(
                train_idx, y, groups, protocol, seed + fold
            )
            selector = SynergyXRegressor(
                drug_dim=drug_dim,
                cell_dim=cell_dim,
                seed=seed + fold,
                epochs=epochs,
                batch_size=256,
                learning_rate=0.001,
                weight_decay=1e-4,
                patience=12,
                min_delta=1e-4,
            )
            selector.fit(
                X[inner_train],
                y[inner_train],
                validation_data=(X[inner_valid], y[inner_valid]),
            )
            selected_epochs = max(1, int(selector.best_epoch_ or epochs))

            model = SynergyXRegressor(
                drug_dim=drug_dim,
                cell_dim=cell_dim,
                seed=seed + fold,
                epochs=selected_epochs,
                batch_size=256,
                learning_rate=0.001,
                weight_decay=1e-4,
                patience=max(12, selected_epochs),
                min_delta=1e-4,
            )
            model.fit(X[train_idx], y[train_idx])
            prediction = np.asarray(model.predict(X[test_idx]), dtype=float)

            masks = [
                ("all", np.ones(len(test_idx), dtype=bool)),
                ("current_drs_available", drs_presence[test_idx]),
                ("current_drs_missing", ~drs_presence[test_idx]),
            ]
            for coverage_group, mask in masks:
                if not mask.any():
                    continue
                selected_idx = test_idx[mask]
                results.append(
                    {
                        "protocol": protocol,
                        "feature_source": "paper_release_features_project_synergyx_attention",
                        "coverage_group": coverage_group,
                        "model": "synergyx_attention",
                        "repeat": repeat + 1,
                        "fold": fold,
                        "test_rows": int(len(selected_idx)),
                        "test_pairs": int(len(set(groups[selected_idx].tolist()))),
                        "selected_epochs": selected_epochs,
                        "device": model.device_name,
                        **_metrics(y[selected_idx], prediction[mask]),
                    }
                )
            print(
                f"{protocol} repeat={repeat + 1} fold={fold}: "
                f"best_epoch={selected_epochs} device={model.device_name} "
                f"test_MSE={_metrics(y[test_idx], prediction)['mse']:.3f}",
                flush=True,
            )

    metrics = pd.DataFrame(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{protocol}_synergyx_attention_r{repeats}"
    fold_path = RESULTS_DIR / f"{tag}_fold_metrics.csv"
    summary_path = RESULTS_DIR / f"{tag}_summary.csv"
    report_path = RESULTS_DIR / f"{tag}_report.json"
    metrics.to_csv(fold_path, index=False)
    summary = (
        metrics.groupby(["protocol", "feature_source", "coverage_group", "model"], as_index=False)
        .agg(
            folds=("fold", "count"),
            rows=("test_rows", "sum"),
            pairs=("test_pairs", "sum"),
            mse=("mse", "mean"),
            mse_sd=("mse", "std"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
            pearson=("pearson", "mean"),
            spearman=("spearman", "mean"),
            selected_epochs=("selected_epochs", "mean"),
        )
    )
    summary.to_csv(summary_path, index=False)
    report = {
        "protocol": protocol,
        "repeats": repeats,
        "base_seed": base_seed,
        "max_epochs": epochs,
        "torch_threads": threads,
        "device": sorted(metrics["device"].dropna().unique().tolist()),
        "coverage": coverage,
        "drs_coverage_on_candidate_rows": drs_counts,
        "model_note": "Project cross-modal attention approximation trained on PCA-compressed author-release features; not the paper authors' exact SynergyX implementation.",
        "selection_note": "Epoch count is selected on an inner validation split; outer fold is used only for evaluation.",
        "paper_internal_mse_reference": 92.16,
        "fold_metrics_file": str(fold_path.relative_to(RESULTS_DIR.parents[1])),
        "summary_file": str(summary_path.relative_to(RESULTS_DIR.parents[1])),
        "elapsed_seconds": time.time() - started,
        "summary": summary.to_dict(orient="records"),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["paper", "pair_grouped"], default="pair_grouped")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=2027)
    args = parser.parse_args()
    run(args.protocol, args.repeats, args.epochs, args.threads, args.base_seed)


if __name__ == "__main__":
    main()
