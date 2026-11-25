"""Nested-tune XGBoost on author-release features with leakage-safe folds."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
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
from ML.storage import load
from ML.training.train import _metrics, _stratification_labels


CONFIGURATIONS = [
    {
        "name": "baseline",
        "n_estimators": 180,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
    },
    {
        "name": "regularized_depth4",
        "n_estimators": 320,
        "max_depth": 4,
        "learning_rate": 0.04,
        "min_child_weight": 5,
        "reg_lambda": 6.0,
    },
    {
        "name": "regularized_depth3",
        "n_estimators": 450,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 4,
        "reg_lambda": 8.0,
    },
]


def _make_xgb(config: dict, seed: int, early_stopping_rounds: int | None = None):
    from xgboost import XGBRegressor

    parameters = {
        key: value
        for key, value in config.items()
        if key != "name"
    }
    return XGBRegressor(
        **parameters,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        verbosity=0,
        early_stopping_rounds=early_stopping_rounds,
    )


def _split_inner(
    train_idx: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    protocol: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
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


def run(protocol: str, repeats: int) -> dict:
    started = time.time()
    frame, X_author, y, groups, coverage = _build_candidate_rows()
    context = load(ML_DIR / "results/models/feature_context.joblib")
    _, _, drs_rows, drs_counts = make_feature_matrix(frame.copy(), "drs", context)
    drs_row_ids = set(drs_rows["_benchmark_row_id"].astype(np.int64).tolist())
    drs_presence = frame["_benchmark_row_id"].astype(np.int64).isin(drs_row_ids).to_numpy()
    project_coverage = {
        "usable_rows": int(drs_presence.sum()),
        "missing_rows": int((~drs_presence).sum()),
        "drs_pair_counts": drs_counts,
        "coverage_note": "DRS coverage is measured against the same candidate rows; the project feature matrix is not rebuilt during nested tuning.",
    }

    labels = _stratification_labels(y, folds=5)
    results: list[dict] = []
    for repeat in range(repeats):
        seed = 2027 + repeat
        if protocol == "pair_grouped":
            splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            splits = splitter.split(X_author, labels, groups)
        else:
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            splits = splitter.split(X_author, labels)

        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            inner_train, inner_valid = _split_inner(
                train_idx, y, groups, protocol, seed + fold
            )
            selected: dict | None = None
            selected_mse = float("inf")
            selected_trees = 0
            selection_rows = []

            for config in CONFIGURATIONS:
                tuner = _make_xgb(config, seed + fold, early_stopping_rounds=25)
                tuner.fit(
                    X_author[inner_train],
                    y[inner_train],
                    eval_set=[(X_author[inner_valid], y[inner_valid])],
                    verbose=False,
                )
                valid_prediction = np.asarray(tuner.predict(X_author[inner_valid]), dtype=float)
                valid_mse = _metrics(y[inner_valid], valid_prediction)["mse"]
                best_trees = int(getattr(tuner, "best_iteration", -1)) + 1
                if best_trees < 1:
                    best_trees = int(config["n_estimators"])
                selection_rows.append(
                    {
                        "configuration": config["name"],
                        "validation_mse": valid_mse,
                        "selected_trees": best_trees,
                    }
                )
                if valid_mse < selected_mse:
                    selected = dict(config)
                    selected_mse = valid_mse
                    selected_trees = best_trees

            if selected is None:
                raise RuntimeError("Nested tuning did not select an XGBoost configuration.")

            selected["n_estimators"] = selected_trees
            model = _make_xgb(selected, seed + fold)
            model.fit(X_author[train_idx], y[train_idx], verbose=False)
            prediction = np.asarray(model.predict(X_author[test_idx]), dtype=float)

            groups_to_score = [("all", np.ones(len(test_idx), dtype=bool))]
            groups_to_score.extend(
                [
                    ("current_drs_available", drs_presence[test_idx]),
                    ("current_drs_missing", ~drs_presence[test_idx]),
                ]
            )
            for coverage_group, mask in groups_to_score:
                if not mask.any():
                    continue
                selected_idx = test_idx[mask]
                results.append(
                    {
                        "protocol": protocol,
                        "feature_source": "paper_release_features_nested_xgboost",
                        "coverage_group": coverage_group,
                        "model": "xgboost_nested",
                        "repeat": repeat + 1,
                        "fold": fold,
                        "test_rows": int(len(selected_idx)),
                        "test_pairs": int(len(set(groups[selected_idx].tolist()))),
                        "selected_configuration": selected["name"],
                        "selected_trees": selected_trees,
                        "inner_validation_mse": selected_mse,
                        **_metrics(y[selected_idx], prediction[mask]),
                    }
                )
            print(
                f"{protocol} repeat={repeat + 1} fold={fold}: "
                f"selected={selected['name']} trees={selected_trees} "
                f"inner_MSE={selected_mse:.3f} "
                f"test_MSE={_metrics(y[test_idx], prediction)['mse']:.3f}",
                flush=True,
            )

    metrics = pd.DataFrame(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_tag = f"{protocol}_nested_xgboost_r{repeats}"
    fold_path = RESULTS_DIR / f"{run_tag}_fold_metrics.csv"
    summary_path = RESULTS_DIR / f"{run_tag}_summary.csv"
    report_path = RESULTS_DIR / f"{run_tag}_report.json"
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
        )
    )
    summary.to_csv(summary_path, index=False)
    report = {
        "protocol": protocol,
        "repeats": repeats,
        "configurations": CONFIGURATIONS,
        "coverage": coverage,
        "project_comparison_rows": project_coverage,
        "nested_selection": "Each outer fold selects a configuration and tree count using only an inner validation split.",
        "evaluation_note": "Pair-grouped rows estimate unseen-pair performance. The paper-style random-row protocol is for comparison only.",
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
    args = parser.parse_args()
    run(args.protocol, args.repeats)


if __name__ == "__main__":
    main()
