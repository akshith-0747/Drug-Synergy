"""Tune DRS regressors on training pairs and evaluate on unseen pair groups.

The test split is reserved before model selection. Candidate XGBoost settings are
ranked only by grouped CV inside the training partition. A SynergyX-inspired
model uses a grouped inner validation split for early stopping. This is an
internal DrugComb holdout, not an independent-study benchmark.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBRegressor

from ML.feature_engineering.features import (
    ML_DIR,
    augment_structure_features,
    build_context,
    make_feature_matrix,
)
from ML.training.train import (
    PROCESSED_FILE,
    _group_validation_indices,
    _metrics,
    _model,
    _pair_groups,
    _stratification_labels,
)

RESULTS_DIR = ML_DIR / "results"
MODEL_DIR = RESULTS_DIR / "models"
OUTER_SEED = 2027
INNER_SEED = 3407
JOBS = 4

XGB_CANDIDATES = {
    "xgboost_baseline": {
        "n_estimators": 180,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "min_child_weight": 1.0,
    },
    "xgboost_depth4_regularized": {
        "n_estimators": 250,
        "max_depth": 4,
        "learning_rate": 0.04,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 2.0,
        "min_child_weight": 2.0,
    },
    "xgboost_depth5_regularized": {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_lambda": 3.0,
        "min_child_weight": 3.0,
    },
    "xgboost_depth3_regularized": {
        "n_estimators": 350,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "reg_lambda": 2.0,
        "min_child_weight": 2.0,
    },
    "xgboost_depth6_strong_regularization": {
        "n_estimators": 250,
        "max_depth": 6,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_lambda": 5.0,
        "min_child_weight": 5.0,
    },
}


def _xgb(params: dict, seed: int) -> XGBRegressor:
    return XGBRegressor(
        **params,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=JOBS,
        random_state=seed,
        verbosity=0,
    )


def _group_splits(X, y, groups, labels, folds: int, seed: int):
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    return list(splitter.split(X, labels, groups))


def run() -> dict:
    context = build_context()
    raw = pd.read_csv(
        PROCESSED_FILE,
        dtype={"block_id": str, "depmap_id": str},
        low_memory=False,
    )
    raw["S"] = pd.to_numeric(raw["S"], errors="coerce")
    raw = raw.dropna(
        subset=["S", "depmap_id", "drug_row", "drug_col"]
    ).reset_index(drop=True)
    augment_structure_features(raw, context)
    X, y, used_rows, coverage = make_feature_matrix(raw, "drs", context)
    groups = _pair_groups(used_rows, context)
    labels = _stratification_labels(y, folds=5)

    outer_splits = _group_splits(X, y, groups, labels, folds=5, seed=OUTER_SEED)
    train_idx, test_idx = outer_splits[0]
    train_groups = groups[train_idx]
    test_groups = groups[test_idx]
    if not set(train_groups).isdisjoint(set(test_groups)):
        raise RuntimeError("Outer holdout shares drug pairs with training.")

    X_train, y_train = X[train_idx], y[train_idx]
    train_labels = _stratification_labels(y_train, folds=3)
    inner_splits = _group_splits(
        X_train,
        y_train,
        train_groups,
        train_labels,
        folds=3,
        seed=INNER_SEED,
    )

    inner_scores = {}
    for name, params in XGB_CANDIDATES.items():
        scores = []
        for fold, (inner_train, inner_valid) in enumerate(inner_splits, start=1):
            model = _xgb(params, seed=OUTER_SEED + fold)
            model.fit(X_train[inner_train], y_train[inner_train])
            predicted = model.predict(X_train[inner_valid])
            scores.append(float(_metrics(y_train[inner_valid], predicted)["rmse"]))
            print(
                f"Inner fold {fold}/3 {name}: RMSE={scores[-1]:.3f}",
                flush=True,
            )
        inner_scores[name] = float(np.mean(scores))

    selected_name = min(inner_scores, key=inner_scores.get)
    selected_params = XGB_CANDIDATES[selected_name]
    print(
        f"Selected on training pairs only: {selected_name} "
        f"(inner mean RMSE={inner_scores[selected_name]:.3f})",
        flush=True,
    )

    candidate_results = []
    fitted_models = {}
    for name, params in (
        ("xgboost_baseline", XGB_CANDIDATES["xgboost_baseline"]),
        ("xgboost_inner_cv_selected", selected_params),
    ):
        if name == "xgboost_inner_cv_selected" and selected_name == "xgboost_baseline":
            model = fitted_models["xgboost_baseline"]
        else:
            model = _xgb(params, seed=OUTER_SEED)
            model.fit(X_train, y_train)
            fitted_models[name] = model
        predicted = np.asarray(model.predict(X[test_idx]), dtype=float)
        candidate_results.append(
            {
                "feature_set": "drs",
                "model": name,
                "selected_xgb_configuration": selected_name,
                "split_strategy": "stratified_grouped_outer_holdout_by_unordered_drug_pair",
                "training_rows": int(len(train_idx)),
                "training_pairs": int(len(set(train_groups))),
                "test_rows": int(len(test_idx)),
                "test_pairs": int(len(set(test_groups))),
                **_metrics(y[test_idx], predicted),
            }
        )

    drug_dim = (X.shape[1] - int(context["cell_dim"])) // 2
    inner_train, inner_valid = _group_validation_indices(
        train_groups,
        fraction=0.15,
        seed=INNER_SEED,
    )
    try:
        tune_synergyx = _model(
            "synergyx",
            seed=OUTER_SEED,
            jobs=JOBS,
            drug_dim=drug_dim,
            cell_dim=int(context["cell_dim"]),
            synergyx_epochs=120,
        )
        tune_synergyx.fit(
            X_train[inner_train],
            y_train[inner_train],
            validation_data=(X_train[inner_valid], y_train[inner_valid]),
        )
        selected_epochs = int(tune_synergyx.best_epoch_ or 120)
        synergyx = _model(
            "synergyx",
            seed=OUTER_SEED,
            jobs=JOBS,
            drug_dim=drug_dim,
            cell_dim=int(context["cell_dim"]),
            synergyx_epochs=selected_epochs,
        )
        synergyx.fit(X_train, y_train)
        synergyx_pred = np.asarray(synergyx.predict(X[test_idx]), dtype=float)
        candidate_results.append(
            {
                "feature_set": "drs",
                "model": "synergyx_inspired_inner_early_stop",
                "selected_xgb_configuration": selected_name,
                "selected_epochs": selected_epochs,
                "split_strategy": "stratified_grouped_outer_holdout_by_unordered_drug_pair",
                "training_rows": int(len(train_idx)),
                "training_pairs": int(len(set(train_groups))),
                "test_rows": int(len(test_idx)),
                "test_pairs": int(len(set(test_groups))),
                **_metrics(y[test_idx], synergyx_pred),
            }
        )
    except ImportError as exc:
        print(f"SynergyX holdout evaluation skipped: {exc}", flush=True)

    result_frame = pd.DataFrame(candidate_results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / "drs_tuning_holdout.csv"
    result_frame.to_csv(result_path, index=False)

    report = {
        "target": "DrugComb S score",
        "feature_set": "drs",
        "coverage": coverage,
        "feature_rows": int(len(y)),
        "unique_drug_pairs": int(len(np.unique(groups))),
        "outer_seed": OUTER_SEED,
        "inner_seed": INNER_SEED,
        "outer_training_rows": int(len(train_idx)),
        "outer_training_pairs": int(len(set(train_groups))),
        "outer_test_rows": int(len(test_idx)),
        "outer_test_pairs": int(len(set(test_groups))),
        "pair_overlap": 0,
        "inner_cv_rmse": inner_scores,
        "selected_xgb_configuration": selected_name,
        "xgb_configurations": XGB_CANDIDATES,
        "results_file": "ML/results/drs_tuning_holdout.csv",
        "validation_scope": "Reserved unordered drug-pair fold; tuning used grouped CV only within the outer training pairs. Rows are from the same DrugComb source, not an independent study.",
    }
    (RESULTS_DIR / "drs_tuning_holdout_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("\nOuter held-out pair results:")
    print(result_frame.to_string(index=False))
    return report


if __name__ == "__main__":
    run()
