"""Refit the CV-selected deployable models on all complete DrugComb rows.

The pair-grouped cross-validation report remains the model-selection record.
This command only refits the selected model family for each view on the full
DrugComb S-score table so prediction uses every row with complete features.
"""
from __future__ import annotations

import gc
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ML.feature_engineering.features import (
    ML_DIR,
    PAIR_SCHEMA_CONCAT,
    augment_structure_features,
    build_context,
    make_feature_matrix,
    make_multimodal_feature_matrix,
)
from ML.models.multilevel import MultilevelStackedRegressor
from ML.storage import dump, load
from ML.training.train import (
    MODEL_DIR,
    PROCESSED_FILE,
    _implementation_for,
    _model,
    _pair_groups,
)

RESULTS_DIR = ML_DIR / "results"
SINGLE_VIEWS = ("structure", "ds", "drs")


def _selected_models() -> dict[str, dict[str, object]]:
    comparison = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    selected = {}
    for view in (*SINGLE_VIEWS, "multilevel"):
        rows = comparison.loc[comparison["feature_set"] == view].copy()
        if "deployment_status" in rows:
            rows = rows.loc[rows["deployment_status"] != "experimental"]
        # Keep research-only pair schemas out of automatic full-data refits
        # unless they are deliberately promoted after a held-out-pair check.
        rows = rows.loc[rows["model"] != "xgboost_pairwise"]
        if rows.empty:
            raise RuntimeError(f"No cross-validation result exists for {view}.")
        rows["rmse"] = pd.to_numeric(rows["rmse"], errors="coerce")
        rows = rows.dropna(subset=["rmse"]).sort_values("rmse")
        if rows.empty:
            raise RuntimeError(f"No valid cross-validation RMSE exists for {view}.")
        best = rows.iloc[0]
        schema = best.get("pair_schema", PAIR_SCHEMA_CONCAT)
        if pd.isna(schema) or not str(schema).strip():
            schema = PAIR_SCHEMA_CONCAT
        recommended_epochs = pd.to_numeric(
            pd.Series([best.get("recommended_epochs", np.nan)]),
            errors="coerce",
        ).iloc[0]
        selected[view] = {
            "model": str(best["model"]),
            "pair_schema": str(schema),
            "recommended_epochs": (
                int(recommended_epochs)
                if np.isfinite(recommended_epochs)
                else 120
            ),
        }
    return selected


def refit() -> dict:
    previous_structure_path = MODEL_DIR / "structure__xgboost.joblib"
    previous_context_path = MODEL_DIR / "feature_context.joblib"
    previous_structure_bundle = (
        load(previous_structure_path) if previous_structure_path.exists() else None
    )
    previous_context = (
        load(previous_context_path) if previous_context_path.exists() else None
    )
    selected_candidates = _selected_models()
    selected = {
        view: candidate["model"]
        for view, candidate in selected_candidates.items()
    }
    selected_schemas = {
        view: candidate["pair_schema"]
        for view, candidate in selected_candidates.items()
    }
    selected_epochs = {
        view: int(candidate["recommended_epochs"])
        for view, candidate in selected_candidates.items()
    }
    context = build_context()
    raw = pd.read_csv(PROCESSED_FILE, dtype={"block_id": str, "depmap_id": str}, low_memory=False)
    raw["S"] = pd.to_numeric(raw["S"], errors="coerce")
    raw = raw.dropna(subset=["S", "depmap_id", "drug_row", "drug_col"]).reset_index(drop=True)
    augment_structure_features(raw, context)

    refit_report = {
        "target": "DrugComb S score",
        "selection_basis": "Lowest mean RMSE in the saved 5-fold pair-grouped CV comparison.",
        "validation_report": "ML/results/model_comparison.csv",
        "full_target_valid_rows": int(len(raw)),
        "selected_models": selected,
        "selected_feature_schemas": selected_schemas,
        "selected_training_epochs": {
            view: selected_epochs[view]
            for view in SINGLE_VIEWS
            if selected[view] == "synergyx"
        },
        "feature_coverage": {},
        "model_files": {},
    }

    for view in SINGLE_VIEWS:
        model_name = selected[view]
        pair_schema = selected_schemas[view]
        X, y, used_rows, coverage = make_feature_matrix(
            raw,
            view,
            context,
            pair_schema=pair_schema,
        )
        if model_name == "stacked_random_forest":
            raise RuntimeError(f"Unexpected stacked model selected for single view {view}.")
        drug_dim = (X.shape[1] - int(context["cell_dim"])) // 2
        model = _model(
            model_name,
            seed=42,
            jobs=-1,
            drug_dim=drug_dim,
            cell_dim=int(context["cell_dim"]),
            synergyx_epochs=selected_epochs[view],
        )
        model.fit(X, y)
        bundle = {
            "model": model,
            "feature_set": view,
            "model_name": model_name,
            "pair_schema": pair_schema,
            "implementation": _implementation_for(model_name),
            "drug_dim": int(drug_dim),
            "cell_dim": int(context["cell_dim"]),
            "n_train_rows": int(len(y)),
            "selected_epochs": selected_epochs[view] if model_name == "synergyx" else None,
            "fit_scope": "all complete rows in the prepared DrugComb S-score table",
            "target": "DrugComb S score",
        }
        if view == "structure" and model_name == "xgboost" and previous_structure_bundle:
            if previous_structure_bundle.get("model_name") == "xgboost":
                legacy_model = previous_structure_bundle["model"]
                legacy_keys = previous_structure_bundle.get("legacy_structure_drug_keys")
                if legacy_keys is None and previous_context:
                    legacy_keys = sorted(previous_context.get("fingerprints", {}).keys())
                if legacy_keys:
                    route_validation_path = (
                        RESULTS_DIR
                        / "pubchem_enrichment_evaluation"
                        / "pubchem_structure_route_holdout.json"
                    )
                    route_validation = (
                        json.loads(route_validation_path.read_text(encoding="utf-8"))
                        if route_validation_path.exists()
                        else None
                    )
                    bundle.update({
                        "model": legacy_model,
                        "expanded_model": model,
                        "structure_routing_enabled": True,
                        "legacy_structure_drug_keys": sorted(set(legacy_keys)),
                        "legacy_model_train_rows": int(
                            previous_structure_bundle.get("n_train_rows", 0)
                        ),
                        "expanded_model_train_rows": int(len(y)),
                        "structure_routing": {
                            "rule": "Keep the legacy XGBoost model when both drugs have legacy fingerprints; use the expanded XGBoost model when PubChem enrichment supplies a previously missing structure.",
                            "validation_report": str(route_validation_path.relative_to(ML_DIR)),
                            "validation": route_validation,
                        },
                    })
        path = MODEL_DIR / f"{view}__{model_name}.joblib"
        dump(bundle, path, compress=3)
        refit_report["feature_coverage"][view] = {
            **coverage,
            "unique_drug_pairs": int(len(np.unique(_pair_groups(used_rows, context)))),
        }
        refit_report["model_files"][view] = path.name
        if bundle.get("structure_routing_enabled"):
            refit_report["structure_routing"] = bundle["structure_routing"]
        print(f"Refit {view}/{model_name}: {len(y):,} rows -> {path}", flush=True)
        del X, y, used_rows, model, bundle
        gc.collect()

    if selected["multilevel"] != "stacked_random_forest":
        raise RuntimeError("The saved multilevel selection is not stacked_random_forest.")
    X_multi, y_multi, used_multi, multi_coverage = make_multimodal_feature_matrix(raw, context)
    groups_multi = _pair_groups(used_multi, context)
    view_dimensions = [int(value) for value in multi_coverage["view_dimensions"]]
    model = MultilevelStackedRegressor(
        view_dimensions=view_dimensions,
        seed=42,
        inner_splits=min(3, len(np.unique(groups_multi))),
        jobs=-1,
    )
    model.fit(X_multi, y_multi, groups=groups_multi)
    bundle = {
        "model": model,
        "feature_set": "multilevel",
        "feature_sets": ["structure", "ds", "drs"],
        "model_name": "stacked_random_forest",
        "implementation": model.estimator_backend_ or "unknown",
        "view_dimensions": view_dimensions,
        "n_train_rows": int(len(y_multi)),
        "fit_scope": "all complete rows in the prepared DrugComb S-score table",
        "target": "DrugComb S score",
    }
    path = MODEL_DIR / "multilevel__stacked_random_forest.joblib"
    dump(bundle, path, compress=3)
    refit_report["feature_coverage"]["multilevel"] = {
        **multi_coverage,
        "unique_drug_pairs": int(len(np.unique(groups_multi))),
    }
    refit_report["model_files"]["multilevel"] = path.name
    dump(context, MODEL_DIR / "feature_context.joblib", compress=3)

    refit_report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    report_path = RESULTS_DIR / "full_data_refit_report.json"
    report_path.write_text(json.dumps(refit_report, indent=2), encoding="utf-8")
    print(f"Refit multilevel/stacked_random_forest: {len(y_multi):,} rows -> {path}", flush=True)
    print(f"Saved full-data refit report: {report_path}", flush=True)
    return refit_report


if __name__ == "__main__":
    refit()
