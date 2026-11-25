"""Promote the PRISM-backed DRS SynergyX after its paired CV gate passes."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ML.feature_engineering.features import ML_DIR, build_context, make_feature_matrix
from ML.storage import dump
from ML.training.train import MODEL_DIR, PROCESSED_FILE, _model, _pair_groups

RESULTS_DIR = ML_DIR / "results"
VALIDATION_DIRS = (
    RESULTS_DIR / "prism_drs_candidate_pair_cv",
    RESULTS_DIR / "prism_drs_candidate_pair_cv_r2",
    RESULTS_DIR / "prism_drs_candidate_pair_cv_19ep",
    RESULTS_DIR / "prism_drs_candidate_pair_cv_nested",
    RESULTS_DIR / "prism_drs_candidate_pair_cv_nested_r2",
)
PROMOTION_REPORT = RESULTS_DIR / "prism_drs_promotion_report.json"
CORE_METRICS = ("rmse", "r2", "spearman", "auc_s_ge_10", "f1_s_ge_10")


def _read_and_validate_runs() -> list[dict]:
    validations = []
    for directory in VALIDATION_DIRS:
        report_path = directory / "run_report.json"
        comparison_path = directory / "model_comparison.csv"
        if not report_path.exists() or not comparison_path.exists():
            raise FileNotFoundError(f"Missing paired validation output under {directory}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("validation", {}).get("same_test_groups_for_both_candidates") is not True:
            raise RuntimeError(f"Validation did not confirm matched held-out drug pairs: {directory}")
        table = pd.read_csv(comparison_path)
        synergyx = table.loc[table["model"].eq("synergyx")].set_index("evaluation_slice")
        required = {"current_coverage", "expanded_model_on_current_coverage"}
        if not required.issubset(set(synergyx.index)):
            raise RuntimeError(f"Missing SynergyX current/expanded metrics in {comparison_path}")
        current = synergyx.loc["current_coverage"]
        expanded = synergyx.loc["expanded_model_on_current_coverage"]
        metric_deltas = {
            metric: float(expanded[metric] - current[metric])
            for metric in CORE_METRICS
        }
        failures = [
            metric
            for metric in CORE_METRICS
            if not np.isfinite(metric_deltas[metric])
            or (metric_deltas[metric] >= 0 if metric == "rmse" else metric_deltas[metric] <= 0)
        ]
        if failures:
            raise RuntimeError(
                f"PRISM DRS failed the prespecified promotion metrics in {directory.name}: "
                + ", ".join(failures)
            )
        fold_path = directory / "fold_metrics.csv"
        fold_table = pd.read_csv(fold_path) if fold_path.exists() else pd.DataFrame()
        if "expanded_selected_epochs" in fold_table:
            selected_epochs = pd.to_numeric(
                fold_table["expanded_selected_epochs"], errors="coerce"
            ).dropna().to_numpy(dtype=int)
            recommended_epochs = int(np.ceil(np.median(selected_epochs))) if len(selected_epochs) else None
        else:
            recommended_epochs = int(report["validation"].get("synergyx_epochs", 120))
        validations.append({
            "directory": str(directory.relative_to(ML_DIR.parent)),
            "seed": int(report["validation"].get("seed", 2027)),
            "synergyx_epochs": int(report["validation"].get("synergyx_epochs", 120)),
            "nested_early_stopping": bool(
                report["validation"].get("synergyx_inner_early_stop", False)
            ),
            "recommended_epochs": recommended_epochs,
            "rows": int(current["rows"]),
            "newly_covered_rows_evaluated": int(
                synergyx.loc["newly_covered_pairs", "rows"]
            ),
            "current_metrics": {metric: float(current[metric]) for metric in CORE_METRICS},
            "expanded_metrics": {metric: float(expanded[metric]) for metric in CORE_METRICS},
            "improvements": metric_deltas,
            "recall_change": float(
                expanded["recall_s_ge_10"] - current["recall_s_ge_10"]
            ),
        })
    return validations


def refit() -> dict:
    validations = _read_and_validate_runs()
    previous_refit = json.loads(
        (RESULTS_DIR / "full_data_refit_report.json").read_text(encoding="utf-8")
    )
    nested_epochs = [
        int(item["recommended_epochs"])
        for item in validations
        if item["nested_early_stopping"] and item["recommended_epochs"] is not None
    ]
    if not nested_epochs:
        raise RuntimeError("No nested pair-grouped early-stopping epoch selection is available.")
    epochs = int(np.ceil(np.median(nested_epochs)))

    context = build_context(include_prism_drs=True)
    raw = pd.read_csv(
        PROCESSED_FILE,
        dtype={"block_id": str, "depmap_id": str},
        low_memory=False,
    )
    raw["S"] = pd.to_numeric(raw["S"], errors="coerce")
    raw = raw.dropna(subset=["S", "depmap_id", "drug_row", "drug_col"]).reset_index(drop=True)
    X, y, used_rows, coverage = make_feature_matrix(raw, "drs", context)
    pair_count = int(len(np.unique(_pair_groups(used_rows, context))))
    model = _model(
        "synergyx",
        seed=42,
        jobs=-1,
        drug_dim=(X.shape[1] - int(context["cell_dim"])) // 2,
        cell_dim=int(context["cell_dim"]),
        synergyx_epochs=epochs,
    )
    model.fit(X, y)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = MODEL_DIR / "archive" / f"before_prism_drs_refit_{timestamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "drs__synergyx.joblib"
    context_path = MODEL_DIR / "feature_context.joblib"
    for source in (model_path, context_path):
        if source.exists():
            shutil.copy2(source, archive_dir / source.name)

    promotion_summary = {
        "promoted_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "DrugComb v1.4 S score",
        "model": "SynergyX-inspired DRS regressor",
        "validation_scope": "Five paired 5-fold runs, including two nested pair-grouped early-stopping repeats; 20,000 target-stratified sampled rows per run",
        "validation_runs": validations,
        "coverage": {
            "baseline_gdsc_rows_in_full_target": 41266,
            "expanded_prism_fallback_rows_in_full_target": 70637,
            "additional_rows": 29371,
            "additional_rows_percent": 71.17,
            "full_refit_feature_rows": int(len(y)),
            "feature_columns": int(X.shape[1]),
            "unique_drug_pairs": pair_count,
            "feature_exclusions": coverage,
        },
        "drs_source": context["drs_source"],
        "drs_source_audit": context.get("drs_source_audit", {}),
        "selected_epochs": epochs,
        "epoch_selection_method": "ceil of the median expanded-model best epoch selected on inner pair-held-out validation across ten outer-training folds from two repeats",
        "backup_directory": str(archive_dir.relative_to(ML_DIR.parent)),
        "caveat": "Held-out DrugComb drug-pair results; not independent-study validation. Core regression and ranking metrics improved in each paired run, while some threshold-specific metrics varied slightly across splits.",
    }
    bundle = {
        "model": model,
        "feature_set": "drs",
        "model_name": "synergyx",
        "pair_schema": "concat_v1",
        "implementation": "pytorch",
        "deployment_status": "production",
        "drug_dim": int((X.shape[1] - context["cell_dim"]) // 2),
        "cell_dim": int(context["cell_dim"]),
        "n_train_rows": int(len(y)),
        "fit_scope": "all complete DrugComb S-score rows supported by GDSC, with quality-filtered PRISM fallback signatures for drugs missing GDSC features",
        "selected_epochs": epochs,
        "epoch_selection_method": "inner pair-grouped validation within the training data",
        "target": "DrugComb S score",
        "drs_source": context["drs_source"],
        "drs_source_audit": context.get("drs_source_audit", {}),
        "validation_report": str(PROMOTION_REPORT.relative_to(ML_DIR.parent)),
    }
    dump(context, context_path, compress=3)
    dump(bundle, model_path, compress=3)
    PROMOTION_REPORT.write_text(json.dumps(promotion_summary, indent=2), encoding="utf-8")

    full_refit = previous_refit
    full_refit.setdefault("selected_models", {})["drs"] = "synergyx"
    full_refit.setdefault("selected_training_epochs", {})["drs"] = epochs
    full_refit.setdefault("model_files", {})["drs"] = model_path.name
    full_refit.setdefault("model_training_rows", {})["drs"] = int(len(y))
    full_refit.setdefault("feature_coverage", {})["drs"] = {
        "input_rows": int(len(raw)),
        "usable_rows": int(len(y)),
        "unique_drug_pairs": pair_count,
        "drs_source": context["drs_source"],
        "source_audit": context.get("drs_source_audit", {}),
    }
    full_refit["drs_selection_note"] = (
        "PRISM was added as a quality-filtered fallback only for drugs without GDSC signatures. "
        "Across five paired group-held-out runs, including two nested early-stopping repeats, the expanded "
        "SynergyX improved RMSE, R2, Spearman, AUC, and F1 on GDSC-covered rows and added 29,371 usable "
        "target rows. Individual threshold metrics can trade off; independent-study validation remains outstanding."
    )
    full_refit["drs_refit_utc"] = datetime.now(timezone.utc).isoformat()
    (RESULTS_DIR / "full_data_refit_report.json").write_text(
        json.dumps(full_refit, indent=2), encoding="utf-8"
    )

    print(
        f"Promoted PRISM-fallback DRS SynergyX: {len(y):,} rows, "
        f"{epochs} epochs. Backup: {archive_dir}",
        flush=True,
    )
    return promotion_summary


if __name__ == "__main__":
    refit()
