"""Assemble completed DS/DRS fits with the separately completed Structure run."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from ML.feature_engineering.features import ML_DIR


MAIN_RUN = ML_DIR / "results" / "drugcomb_v1_5_ssum_all_models_full_v1_pair_cv"
STRUCTURE_RUN = ML_DIR / "results" / "drugcomb_v1_5_ssum_structure_full_v1_pair_cv"
DATASET_LABEL = "DrugComb v1.5 · S_sum · full-data all-model comparison"
TARGET_DESCRIPTION = "DrugComb v1.5 S_sum"
ROWS = {"ds": 142_024, "drs": 93_840}
FAMILIES = ("xgboost", "random_forest", "adaboost", "synergyx", "lasso")

# Captured from the completed two-fold output of the full-data DS/DRS run.
# Values are rounded exactly as printed by the trainer; unavailable metrics
# are left blank rather than reconstructed from rounded console output.
FOLD_SCORES = {
    "ds": {
        "xgboost": [(21.014, 19.920, 0.449), (20.807, 19.792, 0.455)],
        "random_forest": [(20.463, 19.369, 0.478), (20.783, 19.788, 0.456)],
        "adaboost": [(26.684, 25.523, 0.112), (26.504, 25.384, 0.115)],
        "synergyx": [(14.595, 13.749, 0.734), (14.036, 13.455, 0.752)],
        "lasso": [(667990.995, 87.831, -556566893.935), (1305613.331, 89.261, -2147638441.023)],
    },
    "drs": {
        "xgboost": [(22.418, 21.331, 0.422), (22.492, 21.221, 0.426)],
        "random_forest": [(21.563, 20.476, 0.465), (21.916, 20.658, 0.455)],
        "adaboost": [(27.584, 26.456, 0.125), (27.946, 26.684, 0.114)],
        "synergyx": [(16.470, 15.691, 0.688), (17.067, 16.110, 0.670)],
        "lasso": [(15256.057, 93.192, -267773.935), (708.333, 87.203, -568.112)],
    },
}
IMPLEMENTATION = {
    "xgboost": "xgboost",
    "random_forest": "scikit-learn",
    "adaboost": "scikit-learn",
    "synergyx": "pytorch",
    "lasso": "scikit-learn SGDRegressor (L1-penalized least squares)",
}


def _confidence(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, mean
    deviation = pd.Series(values, dtype=float).std(ddof=1)
    margin = 1.96 * float(deviation) / (len(values) ** 0.5)
    return mean - margin, mean + margin


def _logged_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = []
    summaries = []
    for view, models in FOLD_SCORES.items():
        for family, values in models.items():
            clipped = []
            rmse_values = []
            r2_values = []
            for fold, (rmse, clip3_rmse, r2) in enumerate(values, start=1):
                rmse_values.append(rmse)
                clipped.append(clip3_rmse)
                r2_values.append(r2)
                folds.append({
                    "feature_set": view,
                    "model": family,
                    "pair_schema": "concat_v1",
                    "implementation": IMPLEMENTATION[family],
                    "split_strategy": "stratified_group_cv_by_unordered_drug_pair",
                    "dataset_label": DATASET_LABEL,
                    "target_description": TARGET_DESCRIPTION,
                    "target_clip_sigma": 3.0,
                    "rows_evaluated": ROWS[view],
                    "repeat": 1,
                    "fold": fold,
                    "rmse": rmse,
                    "clip3_rmse": clip3_rmse,
                    "r2": r2,
                })
            rmse_low, rmse_high = _confidence(rmse_values)
            clip_low, clip_high = _confidence(clipped)
            r2_low, r2_high = _confidence(r2_values)
            summaries.append({
                "feature_set": view,
                "model": family,
                "pair_schema": "concat_v1",
                "implementation": IMPLEMENTATION[family],
                "deployment_status": "cv_candidate",
                "split_strategy": "stratified_group_cv_by_unordered_drug_pair",
                "dataset_label": DATASET_LABEL,
                "target_description": TARGET_DESCRIPTION,
                "target_clip_sigma": 3.0,
                "folds_evaluated": len(values),
                "rows_evaluated": ROWS[view],
                "rmse": sum(rmse_values) / len(rmse_values),
                "rmse_ci95_low": rmse_low,
                "rmse_ci95_high": rmse_high,
                "clip3_rmse": sum(clipped) / len(clipped),
                "clip3_rmse_ci95_low": clip_low,
                "clip3_rmse_ci95_high": clip_high,
                "r2": sum(r2_values) / len(r2_values),
                "r2_ci95_low": r2_low,
                "r2_ci95_high": r2_high,
            })
    return pd.DataFrame(folds), pd.DataFrame(summaries)


def assemble() -> None:
    structure_report_path = STRUCTURE_RUN / "full_data_refit_report.json"
    if not structure_report_path.exists():
        structure_report_path = STRUCTURE_RUN / "run_report.json"
    structure_summary_path = STRUCTURE_RUN / "model_comparison.csv"
    structure_folds_path = STRUCTURE_RUN / "cv_fold_metrics.csv"
    if not all(path.exists() for path in (structure_report_path, structure_summary_path, structure_folds_path)):
        raise FileNotFoundError("The complete Structure run is required before assembling the comparison.")

    structure_report = json.loads(structure_report_path.read_text(encoding="utf-8"))
    structure_models_dir = STRUCTURE_RUN / "models"
    main_models_dir = MAIN_RUN / "models"
    main_models_dir.mkdir(parents=True, exist_ok=True)
    structure_context = structure_models_dir / "feature_context.joblib"
    main_context = main_models_dir / "feature_context.joblib"
    if not main_context.exists():
        shutil.copy2(structure_context, main_context)
    for family in FAMILIES:
        source = structure_models_dir / f"structure__{family}.joblib"
        if not source.exists():
            raise FileNotFoundError(f"Missing full-data Structure model: {source.name}")
        shutil.copy2(source, main_models_dir / source.name)

    dsdrs_folds, dsdrs_summary = _logged_rows()
    structure_folds = pd.read_csv(structure_folds_path)
    structure_summary = pd.read_csv(structure_summary_path)
    all_folds = pd.concat([dsdrs_folds, structure_folds], ignore_index=True, sort=False)
    all_summary = pd.concat([dsdrs_summary, structure_summary], ignore_index=True, sort=False)
    all_folds.to_csv(MAIN_RUN / "cv_fold_metrics.csv", index=False)
    all_summary.to_csv(MAIN_RUN / "model_comparison.csv", index=False)

    report = structure_report
    coverage = report.setdefault("feature_coverage", {})
    for view, rows in ROWS.items():
        coverage[view] = {
            "input_rows": int(report.get("prepared_rows", 426_956)),
            "usable_rows": rows,
            "feature_rows": rows,
            "feature_columns": 768,
            "validation_group": "unordered drug pair",
        }
    report.update({
        "dataset_label": DATASET_LABEL,
        "target_description": TARGET_DESCRIPTION,
        "max_rows_requested": 0,
        "folds": 2,
        "repeats": 1,
        "feature_sets": ["ds", "drs", "structure"],
        "models_requested": list(FAMILIES),
        "models_trained": list(FAMILIES),
        "models_skipped": [],
        "model_configuration_by_feature": {
            "ds": {
                "random_forest": {"n_estimators": 64, "max_depth": 14},
                "adaboost": {"n_estimators": 80, "base_tree_max_depth": 5},
                "synergyx": {"max_epochs": 12},
                "lasso": {"solver": "SGDRegressor", "alpha": 0.01, "max_iter": 300, "target_scaled": False},
            },
            "drs": {
                "random_forest": {"n_estimators": 64, "max_depth": 14},
                "adaboost": {"n_estimators": 80, "base_tree_max_depth": 5},
                "synergyx": {"max_epochs": 12},
                "lasso": {"solver": "SGDRegressor", "alpha": 0.01, "max_iter": 300, "target_scaled": False},
            },
            "structure": {
                "random_forest": {"n_estimators": 8, "max_depth": 8},
                "adaboost": {"n_estimators": 12, "base_tree_max_depth": 2, "base_tree_max_features": "sqrt"},
                "synergyx": {"max_epochs": 8},
                "lasso": {"solver": "SGDRegressor", "alpha": 0.001, "max_iter": 1000, "target_scaled": True},
            },
        },
        "output_directory": str(MAIN_RUN),
        "note": "DS and DRS fold metrics were retained from the completed full-data trainer output at printed precision; other metrics not captured from those logs are left blank. Structure metrics come from its isolated full-data two-fold run. Hyperparameters differ by view because the full Structure matrix is substantially larger. LASSO showed extreme held-out errors and is not exposed for prediction.",
    })
    (MAIN_RUN / "full_data_refit_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "rows": ROWS | {"structure": 426_956},
        "models_per_view": list(FAMILIES),
        "comparison_csv": str(MAIN_RUN / "model_comparison.csv"),
    }, indent=2))


if __name__ == "__main__":
    assemble()
