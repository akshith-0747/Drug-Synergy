"""Publish completed full-data v1.5 comparison models as selectable candidates.

This never replaces the active XGBoost bundles. Older same-name estimator
bundles are archived before a full-data candidate is installed.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from ML.feature_engineering.features import ML_DIR
from ML.storage import dump, load


DEFAULT_RUN = ML_DIR / "results" / "drugcomb_v1_5_ssum_all_models_full_v1_pair_cv"
ACTIVE_DIR = ML_DIR / "results" / "models"
MODEL_FAMILIES = ("lasso", "random_forest", "adaboost", "synergyx")
FEATURE_VIEWS = ("structure", "ds", "drs")
DATASET_VERSION = "drugcomb_v1_5_ssum"
CONTEXT_NAME = "feature_context_v1_5_allmodels.joblib"


def _native_metrics(row: pd.Series, report: dict) -> dict:
    metrics: dict[str, float | int | str] = {}
    for key, value in row.items():
        if pd.isna(value):
            continue
        if isinstance(value, (int, float)) or str(value).replace(".", "", 1).isdigit():
            metrics[str(key)] = float(value)
        elif key in {"split_strategy", "deployment_status", "implementation"}:
            metrics[str(key)] = str(value)
    clip_sigma = float(report.get("target_clip_sigma") or 0)
    selection_key = f"clip{clip_sigma:g}_rmse" if clip_sigma > 0 else "rmse"
    if selection_key not in metrics and "rmse" in metrics:
        selection_key = "rmse"
    if selection_key in metrics:
        metrics["selection_metric"] = selection_key
        metrics["selection_rmse"] = float(metrics[selection_key])
    return metrics


def promote(run_dir: Path) -> dict:
    report_path = run_dir / "full_data_refit_report.json"
    comparison_path = run_dir / "model_comparison.csv"
    run_model_dir = run_dir / "models"
    if not report_path.exists() or not comparison_path.exists():
        raise FileNotFoundError("The full-data run has not completed its report files yet.")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if int(report.get("max_rows_requested", 0)) != 0:
        raise ValueError("Refusing to publish: this run was not configured for all rows.")
    dataset_label = str(report.get("dataset_label", "")).lower()
    if "drugcomb v1.5" not in dataset_label and "drugcomb_v1_5" not in dataset_label:
        raise ValueError("Refusing to publish: the run is not labeled as DrugComb v1.5.")

    comparison = pd.read_csv(comparison_path)
    active_context = ACTIVE_DIR / CONTEXT_NAME
    source_context = run_model_dir / "feature_context.joblib"
    if not source_context.exists():
        raise FileNotFoundError("The run's feature context is missing.")
    if not active_context.exists():
        dump(load(source_context), active_context, compress=3)

    archived_dir = ML_DIR / "results" / "models_archive" / (
        "pre_full_data_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    published: list[str] = []
    skipped: list[str] = []

    for view in FEATURE_VIEWS:
        coverage = report.get("feature_coverage", {}).get(view, {})
        expected_rows = int(coverage.get("feature_rows", 0))
        current_xgb_path = ACTIVE_DIR / f"{view}__xgboost.joblib"
        try:
            current_xgb = load(current_xgb_path)
        except Exception:
            current_xgb = {}

        for family in MODEL_FAMILIES:
            key = (comparison["feature_set"] == view) & (comparison["model"] == family)
            rows = comparison.loc[key]
            source_model = run_model_dir / f"{view}__{family}.joblib"
            if rows.empty or not source_model.exists():
                skipped.append(f"{view}/{family}: no completed full-data CV model")
                continue
            metric_row = rows.iloc[0]
            required_folds = int(report.get("folds", 0)) * int(report.get("repeats", 0))
            if int(metric_row.get("folds_evaluated", 0)) < required_folds:
                skipped.append(f"{view}/{family}: incomplete pair-grouped validation")
                continue

            bundle = load(source_model)
            if int(bundle.get("n_train_rows", -1)) != expected_rows:
                skipped.append(f"{view}/{family}: final model row count does not match coverage audit")
                continue

            validation_metrics = _native_metrics(metric_row, report)
            if family == "lasso" and validation_metrics.get("selection_rmse", float("inf")) > 50:
                skipped.append(f"{view}/{family}: unstable full-data fit; not exposed for prediction")
                continue

            bundle.update({
                "dataset_version": DATASET_VERSION,
                "deployment_status": "candidate",
                "context_file": CONTEXT_NAME,
                "validation_metrics": validation_metrics,
                "validation_protocol": report.get("validation_protocol"),
                "selection_group": report.get("grouping_key"),
            })
            if int(current_xgb.get("n_train_rows", -2)) == expected_rows:
                for field in ("n_train_cell_lines", "training_cell_line_ids"):
                    if field in current_xgb:
                        bundle[field] = current_xgb[field]

            destination = ACTIVE_DIR / f"{view}__{family}.joblib"
            if destination.exists():
                try:
                    old_bundle = load(destination)
                except Exception:
                    old_bundle = {}
                if old_bundle.get("dataset_version") != DATASET_VERSION:
                    archived_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(destination), str(archived_dir / destination.name))
            dump(bundle, destination, compress=3)
            published.append(
                f"{view}/{family}: {expected_rows:,} rows, "
                f"{bundle['validation_metrics'].get('selection_metric', 'RMSE')}="
                f"{bundle['validation_metrics'].get('selection_rmse', float('nan')):.3f}"
            )

    return {"published": published, "skipped": skipped, "archive": str(archived_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    result = promote(args.run_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
