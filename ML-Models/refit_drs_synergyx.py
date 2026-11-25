"""Promote and refit the confirmed SynergyX-inspired DRS model."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ML.feature_engineering.features import (
    ML_DIR,
    augment_structure_features,
    build_context,
    make_feature_matrix,
)
from ML.storage import dump
from ML.training.train import PROCESSED_FILE, _model

RESULTS_DIR = ML_DIR / "results"
MODEL_DIR = RESULTS_DIR / "models"
CONFIRMATION_FILE = RESULTS_DIR / "drs_candidate_confirmation_folds.csv"
CONFIRMATION_REPORT = RESULTS_DIR / "drs_candidate_confirmation_report.json"
PAIR_SCHEMA = "concat_v1"


def _read_confirmation() -> tuple[pd.DataFrame, int]:
    folds = pd.read_csv(CONFIRMATION_FILE)
    report = json.loads(CONFIRMATION_REPORT.read_text(encoding="utf-8"))
    if report.get("pair_overlap") != 0 or report.get("initial_selection_fold_excluded") is not True:
        raise RuntimeError("DRS candidate confirmation did not pass the pair-held-out checks.")
    if report.get("synergyx_wins_over_baseline", 0) < 3:
        raise RuntimeError("SynergyX did not win enough confirmation folds for promotion.")
    epochs = pd.to_numeric(
        folds.loc[
            folds["model"].eq("synergyx_inspired_inner_early_stop"),
            "selected_epochs",
        ],
        errors="coerce",
    ).dropna()
    if epochs.empty:
        raise RuntimeError("No inner-validation SynergyX epochs were recorded.")
    return folds, int(np.ceil(np.median(epochs)))


def _comparison_rows(folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aliases = {
        "xgboost_baseline": "xgboost",
        "xgboost_inner_cv_selected": "xgboost_tuned",
        "synergyx_inspired_inner_early_stop": "synergyx",
    }
    implementations = {
        "xgboost": "xgboost",
        "xgboost_tuned": "xgboost",
        "synergyx": "pytorch",
    }
    metric_names = [
        "mse", "rmse", "r2", "pearson", "spearman", "auc_s_ge_10",
        "accuracy_s_ge_10", "balanced_accuracy_s_ge_10", "precision_s_ge_10",
        "recall_s_ge_10", "f1_s_ge_10", "average_precision_s_ge_10",
    ]
    cv_rows = []
    summaries = []
    fold_aliases = folds.copy()
    fold_aliases["model"] = fold_aliases["model"].map(aliases)
    for model_name, group in fold_aliases.groupby("model"):
        implementation = implementations[model_name]
        base = {
            "feature_set": "drs",
            "model": model_name,
            "pair_schema": PAIR_SCHEMA,
            "implementation": implementation,
            "deployment_status": "experimental" if model_name == "xgboost_tuned" else "cv_candidate",
            "split_strategy": "stratified_group_cv_by_unordered_drug_pair",
            "folds_evaluated": int(group["fold"].nunique()),
            "rows_evaluated": int(group["test_rows"].sum()),
            "unique_pairs_evaluated": int(group["test_pairs"].sum()),
        }
        for metric in metric_names:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            mean = float(np.mean(values)) if len(values) else float("nan")
            margin = (
                float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1 else float("nan")
            )
            base[metric] = mean
            base[f"{metric}_ci95_low"] = mean - margin if np.isfinite(margin) else float("nan")
            base[f"{metric}_ci95_high"] = mean + margin if np.isfinite(margin) else float("nan")
        if model_name == "synergyx":
            epochs = pd.to_numeric(group["selected_epochs"], errors="coerce").dropna()
            base["recommended_epochs"] = int(np.ceil(np.median(epochs)))
        summaries.append(base)
        for _, row in group.iterrows():
            cv_rows.append(
                {
                    **row.to_dict(),
                    "feature_set": "drs",
                    "model": model_name,
                    "pair_schema": PAIR_SCHEMA,
                    "implementation": implementation,
                    "split_strategy": "stratified_group_cv_by_unordered_drug_pair",
                    "repeat": 1,
                    "rows_evaluated": int(group["test_rows"].sum()),
                    "unique_pairs_evaluated": int(group["test_pairs"].sum()),
                    "deployment_status": "experimental" if model_name == "xgboost_tuned" else "cv_candidate",
                }
            )
    return pd.DataFrame(cv_rows), pd.DataFrame(summaries)


def _update_saved_metrics(new_cv: pd.DataFrame, new_summary: pd.DataFrame) -> None:
    cv_path = RESULTS_DIR / "cv_fold_metrics.csv"
    old_cv = pd.read_csv(cv_path)
    old_cv = old_cv.loc[
        ~((old_cv["feature_set"].eq("drs")) & old_cv["model"].isin(["xgboost", "synergyx"]))
    ].copy()
    for column in set(old_cv.columns).union(new_cv.columns):
        if column not in old_cv:
            old_cv[column] = np.nan
        if column not in new_cv:
            new_cv[column] = np.nan
    pd.concat([old_cv, new_cv], ignore_index=True, sort=False).to_csv(cv_path, index=False)

    summary_path = RESULTS_DIR / "model_comparison.csv"
    old_summary = pd.read_csv(summary_path)
    old_summary = old_summary.loc[
        ~((old_summary["feature_set"].eq("drs")) & old_summary["model"].isin(["xgboost", "synergyx"]))
    ].copy()
    for column in set(old_summary.columns).union(new_summary.columns):
        if column not in old_summary:
            old_summary[column] = np.nan
        if column not in new_summary:
            new_summary[column] = np.nan
    combined = pd.concat([old_summary, new_summary], ignore_index=True, sort=False)
    view_order = {"structure": 0, "ds": 1, "drs": 2, "multilevel": 3}
    combined["_view_order"] = combined["feature_set"].map(view_order).fillna(99)
    combined = combined.sort_values(["_view_order", "rmse"]).drop(columns="_view_order")
    combined.to_csv(summary_path, index=False)


def refit() -> dict:
    confirmation, selected_epochs = _read_confirmation()
    new_cv, new_summary = _comparison_rows(confirmation)

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
    expected_rows = int(pd.to_numeric(confirmation["training_rows"], errors="coerce").max())
    if len(y) < expected_rows:
        raise RuntimeError(
            f"Full-data DRS refit has only {len(y):,} rows; confirmation used {expected_rows:,}."
        )

    model = _model(
        "synergyx",
        seed=42,
        jobs=-1,
        drug_dim=(X.shape[1] - int(context["cell_dim"])) // 2,
        cell_dim=int(context["cell_dim"]),
        synergyx_epochs=selected_epochs,
    )
    model.fit(X, y)
    bundle = {
        "model": model,
        "feature_set": "drs",
        "model_name": "synergyx",
        "pair_schema": PAIR_SCHEMA,
        "implementation": "pytorch",
        "deployment_status": "production",
        "drug_dim": int((X.shape[1] - context["cell_dim"]) // 2),
        "cell_dim": int(context["cell_dim"]),
        "n_train_rows": int(len(y)),
        "fit_scope": "all complete rows in the prepared DrugComb S-score table",
        "selected_epochs": selected_epochs,
        "target": "DrugComb S score",
    }
    model_path = dump(bundle, MODEL_DIR / "drs__synergyx.joblib", compress=3)

    _update_saved_metrics(new_cv, new_summary)
    run_report_path = RESULTS_DIR / "run_report.json"
    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
    run_report.setdefault("additional_validation_runs", {})["drs_synergyx_confirmation"] = {
        "rows": int(confirmation["test_rows"].sum() / 3),
        "pairs": int(confirmation.loc[confirmation.model.eq("xgboost_baseline"), "test_pairs"].sum()),
        "confirmation_folds": 4,
        "synergyx_wins_over_xgboost": 4,
        "synergyx_rmse": float(new_summary.loc[new_summary.model.eq("synergyx"), "rmse"].iloc[0]),
        "xgboost_rmse": float(new_summary.loc[new_summary.model.eq("xgboost"), "rmse"].iloc[0]),
        "source": "ML/results/drs_candidate_confirmation_report.json",
    }
    run_report["models_trained"] = sorted(set(run_report.get("models_trained", [])) | {"synergyx", "xgboost_tuned"})
    run_report_path.write_text(json.dumps(run_report, indent=2), encoding="utf-8")

    refit_path = RESULTS_DIR / "full_data_refit_report.json"
    refit_report = json.loads(refit_path.read_text(encoding="utf-8"))
    refit_report.setdefault("selected_models", {})["drs"] = "synergyx"
    refit_report.setdefault("selected_feature_schemas", {})["drs"] = PAIR_SCHEMA
    refit_report.setdefault("selected_training_epochs", {})["drs"] = selected_epochs
    refit_report.setdefault("model_files", {})["drs"] = model_path.name
    refit_report.setdefault("model_training_rows", {})["drs"] = int(len(y))
    refit_report["drs_selection_note"] = (
        "SynergyX-inspired DRS model beat XGBoost in all four confirmation folds "
        "after excluding the fold used for initial selection; refit on all complete DRS rows."
    )
    refit_report["drs_refit_utc"] = datetime.now(timezone.utc).isoformat()
    refit_path.write_text(json.dumps(refit_report, indent=2), encoding="utf-8")

    print(
        f"Saved full-data DRS SynergyX: {len(y):,} rows, "
        f"{selected_epochs} epochs -> {model_path}",
        flush=True,
    )
    print(new_summary[["model", "folds_evaluated", "rows_evaluated", "rmse", "r2", "recommended_epochs"]].to_string(index=False), flush=True)
    return refit_report


if __name__ == "__main__":
    refit()
