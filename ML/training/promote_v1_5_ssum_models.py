"""Promote the validated DrugComb v1.5 S_sum XGBoost models."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ML.feature_engineering.features import ML_DIR, resolve_drug
from ML.storage import dump, load


RESULTS_DIR = ML_DIR / "results"
MODEL_DIR = RESULTS_DIR / "models"
DATA_FILE = ML_DIR / "data" / "processed" / "matched_drugcomb_v1_5_ssum.csv"
SOURCE_DIR = RESULTS_DIR / "drugcomb_v1_5_ssum_full_xgb_pair_cv"
CONTEXT_FILE = "feature_context_v1_5_ssum.joblib"
FEATURE_VIEWS = ("structure", "ds", "drs")


def _validate_candidate() -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    report_path = SOURCE_DIR / "run_report.json"
    comparison_path = SOURCE_DIR / "model_comparison.csv"
    context_path = SOURCE_DIR / "models" / "feature_context.joblib"
    if not all(path.exists() for path in (report_path, comparison_path, context_path, DATA_FILE)):
        raise FileNotFoundError("The prepared v1.5 S_sum data or its full pair-CV candidate is missing.")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    comparison = pd.read_csv(comparison_path)
    data = pd.read_csv(
        DATA_FILE,
        usecols=["S", "S_sum", "depmap_id", "drug_row", "drug_col"],
        low_memory=False,
    )
    if report.get("target_description") != "DrugComb v1.5 S_sum":
        raise RuntimeError("The candidate was not evaluated against the v1.5 S_sum target.")
    validation_text = " ".join(
        str(report.get(key, ""))
        for key in ("grouping_key", "validation_protocol")
    ).lower()
    if report.get("folds", 0) < 3 or "drug pair" not in validation_text:
        raise RuntimeError("Promotion requires at least 3 pair-grouped validation folds.")
    if report.get("target_clip_sigma") != 3.0:
        raise RuntimeError("The candidate report does not document fold-local 3-sigma clipping.")
    if len(data) != 426_956 or data["depmap_id"].nunique() != 172:
        raise RuntimeError("Prepared data coverage differs from the validated 426,956-row / 172-line dataset.")
    if not np.array_equal(data["S"].to_numpy(), data["S_sum"].to_numpy()):
        raise RuntimeError("The prepared prediction target is not exactly the S_sum column.")

    context = load(context_path)
    expression_ids = set(context["expression"])
    drug_a = data["drug_row"].map(lambda value: resolve_drug(value, context))
    drug_b = data["drug_col"].map(lambda value: resolve_drug(value, context))
    cell_ids = data["depmap_id"].fillna("").astype(str)
    feature_line_coverage = {}
    for view, key in (("structure", "fingerprints"), ("ds", "drug_signatures"), ("drs", "resistance_signatures")):
        available = set(context[key])
        mask = cell_ids.isin(expression_ids) & drug_a.isin(available) & drug_b.isin(available)
        cell_line_ids = sorted(data.loc[mask, "depmap_id"].dropna().astype(str).unique())
        feature_line_coverage[view] = {
            "rows": int(mask.sum()),
            "cell_lines": len(cell_line_ids),
            "cell_line_ids": cell_line_ids,
        }

    selected = comparison.loc[comparison["model"].eq("xgboost")].set_index("feature_set")
    if not set(FEATURE_VIEWS).issubset(selected.index):
        raise RuntimeError("The candidate comparison is missing one or more XGBoost feature views.")
    for view in FEATURE_VIEWS:
        row = selected.loc[view]
        if feature_line_coverage[view]["rows"] != int(report["feature_coverage"][view]["usable_rows"]):
            raise RuntimeError(f"The {view} cell-line audit does not match the reported training coverage.")
        required = ["clip3_rmse", "clip3_r2", "clip3_spearman", "clip3_auc_s_ge_10", "clip3_f1_s_ge_10"]
        if not np.isfinite(pd.to_numeric(row[required], errors="coerce")).all():
            raise RuntimeError(f"The {view} candidate has incomplete held-out metrics.")
        if row["clip3_r2"] <= 0 or row["clip3_spearman"] <= 0 or row["clip3_auc_s_ge_10"] <= 0.5:
            raise RuntimeError(f"The {view} candidate did not clear the minimum validation gate.")

        bundle_path = SOURCE_DIR / "models" / f"{view}__xgboost.joblib"
        bundle = load(bundle_path)
        expected_rows = int(report["feature_coverage"][view]["usable_rows"])
        if bundle.get("n_train_rows") != expected_rows:
            raise RuntimeError(f"The {view} model artifact does not match its reported fit coverage.")
        if bundle.get("target") != "DrugComb v1.5 S_sum" or bundle.get("target_clip", {}).get("sigma") != 3.0:
            raise RuntimeError(f"The {view} model artifact target or clipping metadata is inconsistent.")
    return report, comparison, data, feature_line_coverage


def promote() -> dict:
    report, comparison, data, feature_line_coverage = _validate_candidate()
    source_context = SOURCE_DIR / "models" / "feature_context.joblib"
    active_context = MODEL_DIR / CONTEXT_FILE
    manifest_path = MODEL_DIR / "active_model_manifest.json"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    previous_manifest = {}
    if manifest_path.exists():
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_manifest = {}

    already_active = previous_manifest.get("active_dataset") == "DrugComb v1.5 S_sum"
    if already_active:
        for view in FEATURE_VIEWS:
            path = MODEL_DIR / f"{view}__xgboost.joblib"
            try:
                already_active = already_active and load(path).get("dataset_version") == "drugcomb_v1_5_ssum"
            except Exception:
                already_active = False
                break

    archive_dir = None
    archived = []
    if not already_active:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive_dir = MODEL_DIR / "archive" / f"before_v1_5_ssum_promotion_{stamp}"
        archive_dir.mkdir(parents=True, exist_ok=False)
        for view in FEATURE_VIEWS:
            for suffix in (".joblib", ".pkl"):
                path = MODEL_DIR / f"{view}__xgboost{suffix}"
                if path.exists():
                    shutil.copy2(path, archive_dir / path.name)
                    archived.append(path.name)
        if active_context.exists():
            shutil.copy2(active_context, archive_dir / active_context.name)
            archived.append(active_context.name)

    shutil.copy2(source_context, active_context)
    selected = comparison.loc[comparison["model"].eq("xgboost")].set_index("feature_set")
    active_models = {}
    for view in FEATURE_VIEWS:
        bundle = load(SOURCE_DIR / "models" / f"{view}__xgboost.joblib")
        row = selected.loc[view]
        bundle.update({
            "deployment_status": "production",
            "dataset_version": "drugcomb_v1_5_ssum",
            "dataset_label": "DrugComb v1.5 S_sum",
            "target": "DrugComb v1.5 S_sum",
            "context_file": CONTEXT_FILE,
            "n_train_cell_lines": feature_line_coverage[view]["cell_lines"],
            "training_cell_line_ids": feature_line_coverage[view]["cell_line_ids"],
            "validation_metrics": {
                "protocol": "3-fold stratified group cross-validation by unordered drug pair",
                "selection_metric": "RMSE on fold-local 3-sigma clipped target",
                "selection_rmse": float(row["clip3_rmse"]),
                "clip3_rmse": float(row["clip3_rmse"]),
                "clip3_r2": float(row["clip3_r2"]),
                "clip3_spearman": float(row["clip3_spearman"]),
                "clip3_auc_s_ge_10": float(row["clip3_auc_s_ge_10"]),
                "clip3_f1_s_ge_10": float(row["clip3_f1_s_ge_10"]),
                "raw_rmse": float(row["rmse"]),
                "rows_evaluated": int(row["rows_evaluated"]),
                "unique_pairs_evaluated": int(row["unique_pairs_evaluated"]),
            },
            "fit_scope": "All v1.5 S_sum rows with complete features for this view; final fit uses the full view-specific target-clipped dataset.",
        })
        active_path = dump(bundle, MODEL_DIR / f"{view}__xgboost.joblib", compress=3)
        active_models[view] = {
            "file": active_path.name,
            "training_rows": int(bundle["n_train_rows"]),
            "validation_rows": int(row["rows_evaluated"]),
            "validation_clip3_rmse": float(row["clip3_rmse"]),
            "validation_clip3_r2": float(row["clip3_r2"]),
            "validation_clip3_spearman": float(row["clip3_spearman"]),
            "validation_auc_s_ge_10": float(row["clip3_auc_s_ge_10"]),
            "validation_f1_s_ge_10": float(row["clip3_f1_s_ge_10"]),
            "training_cell_lines": feature_line_coverage[view]["cell_lines"],
        }

    manifest = {
        "active_dataset": "DrugComb v1.5 S_sum",
        "dataset_file": str(DATA_FILE.relative_to(ML_DIR.parent)),
        "rows": int(len(data)),
        "columns": int(len(pd.read_csv(DATA_FILE, nrows=0).columns)),
        "cell_lines": int(data["depmap_id"].nunique()),
        "feature_cell_line_coverage": {
            view: {
                "rows": feature_line_coverage[view]["rows"],
                "cell_lines": feature_line_coverage[view]["cell_lines"],
            }
            for view in FEATURE_VIEWS
        },
        "target": "S_sum",
        "score_clipping_sigma": 3.0,
        "validation_source": str(SOURCE_DIR.relative_to(ML_DIR.parent)),
        "validation_folds": int(report["folds"]),
        "validation_repeats": int(report["repeats"]),
        "models": active_models,
        "context_file": CONTEXT_FILE,
        "previous_models_archive": (
            str(archive_dir.relative_to(MODEL_DIR.parent))
            if archive_dir
            else previous_manifest.get("previous_models_archive")
        ),
        "archived_files": (
            archived
            if archive_dir
            else previous_manifest.get("archived_files", [])
        ),
        "note": "The three views predict the same v1.5 S_sum target. Scores are internal pair-held-out validation and are not independent-study validation.",
    }
    manifest["previous_models_archive"] = (
        str(archive_dir.relative_to(MODEL_DIR.parent))
        if archive_dir
        else previous_manifest.get("previous_models_archive")
    )
    manifest["archived_files"] = (
        archived
        if archive_dir
        else previous_manifest.get("archived_files", [])
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    refit_path = RESULTS_DIR / "full_data_refit_report.json"
    if refit_path.exists():
        refit = json.loads(refit_path.read_text(encoding="utf-8"))
    else:
        refit = {}
    refit["active_prediction_profile"] = manifest
    refit_path.write_text(json.dumps(refit, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    promote()
