"""Fit the author-feature XGBoost fallback after its grouped-CV confirmation."""
from __future__ import annotations

import json
from ML.evaluation.paper_release_feature_benchmark import (
    RESULTS_DIR as BENCHMARK_DIR,
    _build_candidate_rows,
    _load_author_features,
)
from ML.feature_engineering.features import (
    ML_DIR,
    _load_drug_maps,
    build_context,
    make_feature_matrix,
)
from ML.storage import dump, load
from ML.models.synergyx import SynergyXRegressor
from ML.training.train import _model


MODEL_DIR = ML_DIR / "results" / "models"
MODEL_PATH = MODEL_DIR / "paper_drs__xgboost.joblib"
CONTEXT_PATH = MODEL_DIR / "feature_context.joblib"
FOLD_METRICS_PATH = BENCHMARK_DIR / "pair_grouped_structure_xgboost_r3_fold_metrics.csv"
SUPPLEMENTARY_METRICS_PATH = BENCHMARK_DIR / "paper_drs_fallback_model_comparison.csv"
ATTENTION_COMPARISON_PATH = BENCHMARK_DIR / "paper_drs_attention_model_comparison.csv"


def refit() -> dict:
    frame, X, y, groups, metadata = _build_candidate_rows()
    if not FOLD_METRICS_PATH.exists():
        raise FileNotFoundError(
            "The three-seed pair-grouped fallback benchmark is missing; "
            "run the paper feature benchmark before fitting this fallback."
        )
    import pandas as pd

    fold_metrics = pd.read_csv(FOLD_METRICS_PATH)
    cohort_metrics = fold_metrics[
        fold_metrics["coverage_group"].eq("current_drs_missing")
    ]
    paired = cohort_metrics.pivot(
        index=["repeat", "fold"],
        columns="feature_source",
        values="mse",
    )
    if (
        len(paired) != 15
        or not {"paper_release_features", "project_structure"}.issubset(paired.columns)
        or not (paired["paper_release_features"] < paired["project_structure"]).all()
    ):
        raise RuntimeError(
            "The author-feature fallback did not beat structure XGBoost "
            "in all 15 matched DRS-missing folds; refusing to publish it."
        )

    paper_folds = cohort_metrics[
        cohort_metrics["feature_source"].eq("paper_release_features")
    ]
    validation_summary = paper_folds.mean(numeric_only=True).to_dict()
    context = load(CONTEXT_PATH) if CONTEXT_PATH.exists() else build_context()
    _, _, drs_rows, _ = make_feature_matrix(frame.copy(), "drs", context)
    drs_row_ids = set(drs_rows["_benchmark_row_id"].astype(int).tolist())
    fallback_rows = frame.loc[~frame["_benchmark_row_id"].astype(int).isin(drs_row_ids)]
    fallback_pairs = {
        "|".join(sorted(pair))
        for pair in zip(fallback_rows["_smiles_row"], fallback_rows["_smiles_col"])
    }

    drug_features, cell_features = _load_author_features()
    model = _model("xgboost", 2027, -1)
    model.fit(X, y)

    attention_report_path = BENCHMARK_DIR / "paper_synergyx_attention_r3_report.json"
    grouped_attention_report_path = BENCHMARK_DIR / "pair_grouped_synergyx_attention_r3_report.json"
    if (
        not attention_report_path.exists()
        or not grouped_attention_report_path.exists()
        or not ATTENTION_COMPARISON_PATH.exists()
    ):
        raise FileNotFoundError("Three-seed attention validation reports are required before pair-aware refitting.")
    attention_report = json.loads(attention_report_path.read_text(encoding="utf-8"))
    grouped_attention_report = json.loads(grouped_attention_report_path.read_text(encoding="utf-8"))
    attention_metrics = next(
        row for row in attention_report["summary"] if row["coverage_group"] == "all"
    )
    grouped_attention_metrics = next(
        row for row in grouped_attention_report["summary"] if row["coverage_group"] == "all"
    )
    attention_comparison = pd.read_csv(ATTENTION_COMPARISON_PATH)
    unseen_pair_xgboost_metrics = attention_comparison[
        attention_comparison["feature_set"].eq("paper_drs_unseen_pair")
        & attention_comparison["model"].eq("xgboost")
    ]
    if len(unseen_pair_xgboost_metrics) != 1:
        raise RuntimeError("The matched unseen-pair XGBoost comparison is missing or ambiguous.")
    if attention_metrics["folds"] != 15 or attention_metrics["mse"] >= 92.16:
        raise RuntimeError("The attention model did not beat the paper reference across 15 paper-style folds.")
    if grouped_attention_metrics["folds"] != 15:
        raise RuntimeError("The attention model lacks three-seed unseen-pair validation.")

    selected_attention_epochs = int(round(attention_metrics["selected_epochs"]))
    attention_model = SynergyXRegressor(
        drug_dim=int(metadata["drug_feature_dim"]),
        cell_dim=int(metadata["cell_feature_dim"]),
        seed=42,
        epochs=selected_attention_epochs,
        batch_size=256,
        learning_rate=0.001,
        weight_decay=1e-4,
        patience=max(12, selected_attention_epochs),
        min_delta=1e-4,
    )
    attention_model.fit(X, y)

    bundle = {
        "model": model,
        "known_pair_model": attention_model,
        "known_pair_model_name": "synergyx_attention",
        "known_pair_keys": sorted(set(str(group) for group in groups.tolist())),
        "pair_routing": {
            "rule": "Use the attention model when the unordered drug pair appeared in fallback training; use XGBoost for unseen pairs.",
            "paper_style_cv_mse": float(attention_metrics["mse"]),
            "paper_reference_mse": 92.16,
            "unseen_pair_attention_mse": float(grouped_attention_metrics["mse"]),
            "unseen_pair_xgboost_mse": float(unseen_pair_xgboost_metrics.iloc[0]["mse"]),
            "paper_validation_report": str(attention_report_path.relative_to(ML_DIR)),
            "pair_grouped_validation_report": str(grouped_attention_report_path.relative_to(ML_DIR)),
        },
        "feature_set": "paper_drs",
        "model_name": "xgboost",
        "implementation": "xgboost",
        "pair_schema": "paper_release_concat_v1",
        "drug_dim": int(metadata["drug_feature_dim"]),
        "cell_dim": int(metadata["cell_feature_dim"]),
        "drug_features": drug_features,
        "cell_features": cell_features,
        "training_rows": int(len(y)),
        "unique_pairs": int(metadata["unique_pairs"]),
        "feature_coverage": metadata,
        "source_arrays": [
            "ML/data/processed/paper_reference/DEGsenWithMASK.npy",
            "ML/data/processed/paper_reference/drugSmile_drugSubEmbed_2644.npy",
            "ML/data/processed/paper_reference/985_cellGraphs_exp_mut_cn_eff_dep_met_4079_genes_norm.npy",
        ],
        "validation_report": (
            "ML/results/paper_feature_benchmark/"
            "pair_grouped_structure_xgboost_r3_report.json"
        ),
        "deployment_status": "production_fallback",
        "pair_routing_enabled": True,
        "validation_metrics": {
            "protocol": "stratified_group_cv_by_unordered_drug_pair",
            "repeats": 3,
            "folds": 5,
            "coverage_group": "current_drs_missing",
            "rows_evaluated": int(len(fallback_rows)),
            "unique_pairs_evaluated": int(len(fallback_pairs)),
            "mse": float(validation_summary["mse"]),
            "rmse": float(validation_summary["rmse"]),
            "r2": float(validation_summary["r2"]),
            "mse_wins_vs_structure": int(
                (paired["paper_release_features"] < paired["project_structure"]).sum()
            ),
            "matched_folds": int(len(paired)),
        },
        "selection_note": (
            "Automatic fallback only when the current GDSC DRS model lacks "
            "features and the author-release drug and cell features are available. "
            "The current DRS model retains priority."
        ),
    }
    saved_path = dump(bundle, MODEL_PATH, compress=3)

    # Keep the canonical drug-to-SMILES map in the shared serving context so
    # paper-release predictions do not need to reread the chemical table.
    context = load(CONTEXT_PATH) if CONTEXT_PATH.exists() else build_context()
    if not context.get("smiles"):
        _, smiles_by_id = _load_drug_maps()
        context["smiles"] = smiles_by_id
        dump(context, CONTEXT_PATH, compress=3)

    metrics_row = {
        "feature_set": "paper_drs",
        "model": "xgboost",
        "implementation": "xgboost",
        "split_strategy": "stratified_group_cv_by_unordered_drug_pair",
        "folds_evaluated": 15,
        "pair_schema": "paper_release_concat_v1",
        "deployment_status": "production_fallback",
        "rows_evaluated": int(len(fallback_rows)),
        "unique_pairs_evaluated": int(len(fallback_pairs)),
        "mse": float(validation_summary["mse"]),
        "rmse": float(validation_summary["rmse"]),
        "r2": float(validation_summary["r2"]),
        "pearson": float(validation_summary["pearson"]),
        "spearman": float(validation_summary["spearman"]),
        "auc_s_ge_10": float(validation_summary["auc_s_ge_10"]),
        "accuracy_s_ge_10": float(validation_summary["accuracy_s_ge_10"]),
        "balanced_accuracy_s_ge_10": float(validation_summary["balanced_accuracy_s_ge_10"]),
        "precision_s_ge_10": float(validation_summary["precision_s_ge_10"]),
        "recall_s_ge_10": float(validation_summary["recall_s_ge_10"]),
        "f1_s_ge_10": float(validation_summary["f1_s_ge_10"]),
        "average_precision_s_ge_10": float(validation_summary["average_precision_s_ge_10"]),
        "validation_scope": "DRS-missing rows with author features",
        "mse_wins_vs_structure": int((paired["paper_release_features"] < paired["project_structure"]).sum()),
        "matched_folds": int(len(paired)),
        "recommended_epochs": None,
    }
    SUPPLEMENTARY_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics_row]).to_csv(SUPPLEMENTARY_METRICS_PATH, index=False)

    report = {
        "model_file": str(saved_path.relative_to(ML_DIR.parent)),
        "feature_set": "paper_drs",
        "model": "xgboost",
        "training_rows": int(len(frame)),
        "unique_pairs": int(metadata["unique_pairs"]),
        "drug_count": int(metadata["unique_drugs"]),
        "cell_line_count": int(metadata["cell_lines"]),
        "feature_dimensions": {
            "drug": int(metadata["drug_feature_dim"]),
            "cell": int(metadata["cell_feature_dim"]),
            "total": int(metadata["total_feature_dim"]),
        },
        "fallback_only": True,
        "pair_routing": bundle["pair_routing"],
        "known_pair_model_training_rows": int(len(y)),
        "known_pair_model_epochs": selected_attention_epochs,
        "validation_report": bundle["validation_report"],
        "validation_metrics": bundle["validation_metrics"],
        "scoreboard_file": str(SUPPLEMENTARY_METRICS_PATH.relative_to(ML_DIR.parent)),
        "paper_internal_mse_reference": 92.16,
        "note": (
            "The pair-aware fallback uses the attention candidate for training-seen drug pairs and XGBoost for unseen pairs. The attention candidate scored below the paper's published MSE on a smaller author-feature-supported cohort under row-wise CV, but this is not an exact comparison; attention did not beat XGBoost on unseen-pair CV and coverage remains limited."
        ),
    }
    report_path = BENCHMARK_DIR / "paper_release_fallback_model.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = refit()
    print(json.dumps(result, indent=2))
