"""Check DRS candidates on grouped folds not used for their first selection."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ML.evaluation.tune_drs_holdout import (
    INNER_SEED,
    JOBS,
    OUTER_SEED,
    RESULTS_DIR,
    XGB_CANDIDATES,
    _xgb,
)
from ML.feature_engineering.features import (
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

SELECTED_XGB = "xgboost_depth6_strong_regularization"


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
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=OUTER_SEED,
    )
    outer_splits = list(splitter.split(X, labels, groups))
    results = []
    evaluated_test_pairs: set[str] = set()

    for fold_number, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        if fold_number == 1:
            continue  # This fold was used for initial model selection.
        train_groups = groups[train_idx]
        test_groups = set(groups[test_idx])
        if not set(train_groups).isdisjoint(test_groups):
            raise RuntimeError(f"Fold {fold_number} shares drug pairs across train/test.")
        if evaluated_test_pairs.intersection(test_groups):
            raise RuntimeError("Confirmation test folds unexpectedly overlap.")
        evaluated_test_pairs.update(test_groups)

        X_train, y_train = X[train_idx], y[train_idx]
        fold_specs = [
            ("xgboost_baseline", XGB_CANDIDATES["xgboost_baseline"]),
            ("xgboost_inner_cv_selected", XGB_CANDIDATES[SELECTED_XGB]),
        ]
        for name, params in fold_specs:
            model = _xgb(params, seed=OUTER_SEED + fold_number)
            model.fit(X_train, y_train)
            scores = _metrics(y[test_idx], model.predict(X[test_idx]))
            results.append(
                {
                    "fold": fold_number,
                    "model": name,
                    "training_rows": int(len(train_idx)),
                    "training_pairs": int(len(set(train_groups))),
                    "test_rows": int(len(test_idx)),
                    "test_pairs": int(len(test_groups)),
                    **scores,
                }
            )

        inner_train, inner_valid = _group_validation_indices(
            train_groups,
            fraction=0.15,
            seed=INNER_SEED + fold_number,
        )
        tuning_model = _model(
            "synergyx",
            seed=OUTER_SEED + fold_number,
            jobs=JOBS,
            drug_dim=(X.shape[1] - int(context["cell_dim"])) // 2,
            cell_dim=int(context["cell_dim"]),
            synergyx_epochs=120,
        )
        tuning_model.fit(
            X_train[inner_train],
            y_train[inner_train],
            validation_data=(X_train[inner_valid], y_train[inner_valid]),
        )
        selected_epochs = int(tuning_model.best_epoch_ or 120)
        synergyx = _model(
            "synergyx",
            seed=OUTER_SEED + fold_number,
            jobs=JOBS,
            drug_dim=(X.shape[1] - int(context["cell_dim"])) // 2,
            cell_dim=int(context["cell_dim"]),
            synergyx_epochs=selected_epochs,
        )
        synergyx.fit(X_train, y_train)
        scores = _metrics(y[test_idx], synergyx.predict(X[test_idx]))
        results.append(
            {
                "fold": fold_number,
                "model": "synergyx_inspired_inner_early_stop",
                "selected_epochs": selected_epochs,
                "training_rows": int(len(train_idx)),
                "training_pairs": int(len(set(train_groups))),
                "test_rows": int(len(test_idx)),
                "test_pairs": int(len(test_groups)),
                **scores,
            }
        )
        print(
            f"Confirmation fold {fold_number}/5: baseline="
            f"{results[-3]['rmse']:.3f}, selected XGB={results[-2]['rmse']:.3f}, "
            f"SynergyX={results[-1]['rmse']:.3f} (epochs={selected_epochs})",
            flush=True,
        )

    frame = pd.DataFrame(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_file = RESULTS_DIR / "drs_candidate_confirmation_folds.csv"
    frame.to_csv(result_file, index=False)
    summary = frame.groupby("model").agg(
        folds=("fold", "nunique"),
        rows_evaluated=("test_rows", "sum"),
        test_pairs=("test_pairs", "sum"),
        rmse_mean=("rmse", "mean"),
        rmse_sd=("rmse", "std"),
        r2_mean=("r2", "mean"),
        spearman_mean=("spearman", "mean"),
    )
    wins = frame.loc[frame.model.isin(["xgboost_baseline", "xgboost_inner_cv_selected", "synergyx_inspired_inner_early_stop"])]
    wins = wins.pivot(index="fold", columns="model", values="rmse")
    synergyx_wins = int(
        (wins["synergyx_inspired_inner_early_stop"] < wins["xgboost_baseline"]).sum()
    )
    report = {
        "coverage": coverage,
        "feature_rows": int(len(y)),
        "total_feature_pairs": int(len(np.unique(groups))),
        "confirmation_folds": [2, 3, 4, 5],
        "confirmation_rows": int(frame.loc[frame.model.eq("xgboost_baseline"), "test_rows"].sum()),
        "confirmation_pairs": len(evaluated_test_pairs),
        "pair_overlap": 0,
        "initial_selection_fold_excluded": True,
        "synergyx_wins_over_baseline": synergyx_wins,
        "summary": summary.reset_index().to_dict(orient="records"),
        "fold_metrics_file": "ML/results/drs_candidate_confirmation_folds.csv",
        "validation_scope": "Four unseen unordered-pair folds from the same DrugComb source; fold 1, used for initial selection, is excluded. This is not independent-study validation.",
    }
    (RESULTS_DIR / "drs_candidate_confirmation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("\nConfirmation summary:")
    print(summary.to_string())
    print(f"SynergyX beat baseline in {synergyx_wins}/4 confirmation folds.")
    return report


if __name__ == "__main__":
    run()
