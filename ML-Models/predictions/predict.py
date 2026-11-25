"""Load a trained model and predict a DrugComb S score for a drug pair."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ML.feature_engineering.features import ML_DIR, normalize, pair_vector, resolve_drug
from ML.feature_engineering.paper_release import canonicalize_smiles, paper_release_pair_vector
from ML.storage import load

MODEL_DIR = ML_DIR / "results" / "models"


def available_models() -> list[dict[str, str]]:
    # Prefer the newest joblib bundle when a previous fallback run left a
    # same-named pickle behind. This also keeps API/UI model lists unique.
    selected: dict[tuple[str, str], Path] = {}
    for suffix in (".joblib", ".pkl"):
        for path in sorted(MODEL_DIR.glob(f"*__*{suffix}")):
            feature_set, model_name = path.stem.split("__", 1)
            if model_name in {"xgboost_pairwise", "xgboost_regularized"}:
                # Research-only XGBoost candidates stay hidden unless explicitly promoted
                # after held-out-pair confirmation.
                try:
                    bundle = load(path)
                except Exception:
                    continue
                if bundle.get("deployment_status") != "production":
                    continue
            if feature_set == "paper_drs":
                try:
                    bundle = load(path)
                except Exception:
                    continue
                if bundle.get("deployment_status") != "production_fallback":
                    continue
            selected.setdefault((feature_set, model_name), path)
    return [
        {"feature_set": feature_set, "model": model_name, "file": path.name}
        for (feature_set, model_name), path in sorted(selected.items())
    ]


def _context_path() -> Path:
    for suffix in (".joblib", ".pkl"):
        path = MODEL_DIR / f"feature_context{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError("Feature context is missing. Run model training first.")


def _full_data_validation_metrics() -> dict[tuple[str, str], dict]:
    """Load the like-for-like full-data comparison used for auto selection."""
    report = (
        ML_DIR
        / "results"
        / "drugcomb_v1_5_ssum_all_models_full_v1_pair_cv"
        / "model_comparison.csv"
    )
    if not report.exists():
        return {}
    try:
        import pandas as pd

        table = pd.read_csv(report)
        required = {"feature_set", "model", "clip3_rmse"}
        if not required.issubset(table.columns):
            return {}
        if "dataset_label" in table.columns:
            table = table.loc[
                table["dataset_label"].fillna("").astype(str).str.startswith(
                    "DrugComb v1.5 · S_sum · full-data all-model comparison"
                )
            ]
        table["clip3_rmse"] = pd.to_numeric(table["clip3_rmse"], errors="coerce")
        rows = {}
        for row in table.dropna(subset=["clip3_rmse"]).to_dict("records"):
            key = (str(row["feature_set"]), str(row["model"]))
            rmse = float(row["clip3_rmse"])
            if not np.isfinite(rmse):
                continue
            rows[key] = {
                "rmse": rmse,
                "metric": "Pair-held-out CV RMSE · 3σ",
                "folds": row.get("folds_evaluated"),
                "source": row.get("dataset_label"),
            }
        return rows
    except Exception:
        return {}


def _rank_choices(
    choices: list[dict[str, str]],
    current_validation: dict[tuple[str, str], dict] | None = None,
) -> list[dict[str, str]]:
    """Rank trained candidates by recorded CV RMSE, with stable fallbacks."""
    scores: dict[tuple[str, str], float] = {}
    selected_models: dict[str, str] = {}
    results_dir = ML_DIR / "results"

    # Use the comparison summary when available, or reconstruct its ranking
    # from fold-level metrics if the summary file is empty or unreadable.
    for filename in ("model_comparison.csv", "cv_fold_metrics.csv"):
        report = results_dir / filename
        if not report.exists():
            continue
        try:
            import pandas as pd

            table = pd.read_csv(report)
            if not {"feature_set", "model", "rmse"}.issubset(table.columns):
                continue
            table["rmse"] = pd.to_numeric(table["rmse"], errors="coerce")
            table = table.dropna(subset=["rmse"])
            if filename == "cv_fold_metrics.csv":
                table = table.groupby(
                    ["feature_set", "model"], as_index=False
                )["rmse"].mean()
            for row in table.itertuples(index=False):
                key = (str(row.feature_set), str(row.model))
                scores.setdefault(key, float(row.rmse))
        except Exception:
            continue

    # Use one consistent full-data, pair-held-out comparison for all model
    # families. Bundle metadata remains the fallback when that report is absent.
    current_validation = current_validation or _full_data_validation_metrics()
    for key, metrics in current_validation.items():
        scores[key] = float(metrics["rmse"])

    # If both CV files are unavailable, prefer the model recorded by the
    # completed full-data refit instead of silently defaulting to Random Forest.
    refit_report = results_dir / "full_data_refit_report.json"
    if refit_report.exists():
        try:
            report_data = json.loads(refit_report.read_text(encoding="utf-8"))
            selected_models = {
                str(view): str(model)
                for view, model in report_data.get("selected_models", {}).items()
            }
        except Exception:
            selected_models = {}

    # Promoted versioned bundles carry validation metrics from their own
    # target and dataset. Use those rather than comparing RMSE values across
    # different DrugComb versions or target definitions.
    for item in choices:
        try:
            bundle = load(MODEL_DIR / item["file"])
        except Exception:
            continue
        metrics = bundle.get("validation_metrics") or {}
        rmse = metrics.get("selection_rmse")
        if bundle.get("deployment_status") in {"production", "candidate"} and rmse is not None:
            try:
                numeric_rmse = float(rmse)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric_rmse):
                scores.setdefault((item["feature_set"], item["model"]), numeric_rmse)

    feature_priority = {
        "structure": 0,
        "ds": 1,
        "drs": 2,
        "paper_drs": 3,
        "multilevel": 4,
    }
    model_priority = {
        "random_forest": 0,
        "xgboost": 1,
        "xgboost_pairwise": 2,
        "adaboost": 3,
        "lasso": 4,
        "synergyx": 5,
        "stacked_random_forest": 6,
    }

    return sorted(
        choices,
        key=lambda item: (
            scores.get(
                (item["feature_set"], item["model"]),
                float("inf"),
            ),
            0 if selected_models.get(item["feature_set"]) == item["model"] else 1,
            feature_priority.get(item["feature_set"], 99),
            model_priority.get(item["model"], 99),
            item["model"],
        ),
    )


def _build_vector(
    drug_a: str,
    drug_b: str,
    cell_line: str,
    feature_set: str,
    context: dict,
    bundle: dict,
) -> np.ndarray:
    if feature_set == "multilevel":
        view_sets = bundle.get(
            "feature_sets",
            ["structure", "ds", "drs"],
        )
        return np.concatenate(
            [
                pair_vector(
                    drug_a,
                    drug_b,
                    cell_line,
                    view,
                    context,
                    pair_schema=bundle.get("pair_schemas", {}).get(
                        view,
                        "concat_v1",
                    ),
                )
                for view in view_sets
            ]
        )

    if feature_set == "paper_drs":
        return paper_release_pair_vector(
            drug_a,
            drug_b,
            cell_line,
            context,
            bundle,
        )

    return pair_vector(
        drug_a,
        drug_b,
        cell_line,
        feature_set,
        context,
        pair_schema=bundle.get("pair_schema", "concat_v1"),
    )


def _paper_pair_key(drug_a: str, drug_b: str, context: dict) -> str | None:
    """Return the canonical unordered pair key used by the author-feature bundle."""
    smiles_by_id = context.get("smiles", {})
    canonical_smiles = []
    for drug_name in (drug_a, drug_b):
        drug_id = resolve_drug(drug_name, context)
        smiles = canonicalize_smiles(smiles_by_id.get(drug_id) or drug_name)
        if not smiles:
            return None
        canonical_smiles.append(smiles)
    return "|".join(sorted(canonical_smiles))


def predict(
    drug_a: str,
    drug_b: str,
    cell_line: str,
    feature_set: str = "auto",
    model_name: str | None = None,
) -> dict:
    requested_view = (feature_set or "auto").strip().lower()
    known_views = {"structure", "ds", "drs", "paper_drs", "multilevel"}

    if requested_view != "auto" and requested_view not in known_views:
        raise ValueError(
            "Feature view must be auto, structure, ds, paper_drs, or multilevel."
        )

    choices = available_models()
    if requested_view == "auto":
        # Keep the default website prediction aligned with the paper's three
        # feature views. The custom stack remains available when explicitly
        # requested, but a legacy saved bundle must not enter auto-selection.
        choices = [
            item for item in choices
            if item["feature_set"] != "multilevel"
        ]
    if requested_view != "auto":
        choices = [
            item for item in choices
            if item["feature_set"] == requested_view
        ]

    model_choice = (model_name or "auto").strip().lower()
    automatic_model_selection = model_choice in {"", "auto", "best"}
    effective_model_name = None if automatic_model_selection else model_choice
    if effective_model_name:
        choices = [
            item for item in choices
            if item["model"] == effective_model_name
        ]

    if automatic_model_selection:
        # Auto mode only considers current full-data v1.5 bundles with a
        # recorded pair-held-out validation score. This excludes old sampled
        # experiments and unstable LASSO from user predictions.
        current_choices = []
        for item in choices:
            if (
                item["feature_set"] not in {"structure", "ds", "drs"}
                or item["model"] == "lasso"
            ):
                continue
            try:
                bundle = load(MODEL_DIR / item["file"])
                metrics = bundle.get("validation_metrics") or {}
                selection_rmse = float(metrics.get("selection_rmse"))
            except Exception:
                continue
            if (
                bundle.get("dataset_version") == "drugcomb_v1_5_ssum"
                and bundle.get("deployment_status") in {"production", "candidate"}
                and np.isfinite(selection_rmse)
            ):
                current_choices.append(item)
        choices = current_choices

    if not automatic_model_selection and effective_model_name not in {None, "xgboost"}:
        # Non-XGBoost choices in the website must be full-data DrugComb v1.5
        # bundles. Older sampled bundles remain on disk for reproducibility,
        # but must not silently enter current predictions.
        current_choices = []
        for item in choices:
            try:
                bundle = load(MODEL_DIR / item["file"])
            except Exception:
                continue
            if (
                bundle.get("dataset_version") == "drugcomb_v1_5_ssum"
                and bundle.get("deployment_status") in {"production", "candidate"}
            ):
                current_choices.append(item)
        choices = current_choices

    # Explicit XGBoost requests stay on the production v1.5 profile. Auto mode
    # uses the best validated current model separately for each feature view.
    if not automatic_model_selection and effective_model_name == "xgboost" and requested_view != "paper_drs":
        active_choices = []
        for item in choices:
            try:
                bundle = load(MODEL_DIR / item["file"])
            except Exception:
                continue
            if bundle.get("dataset_version") == "drugcomb_v1_5_ssum":
                active_choices.append(item)
        if active_choices:
            choices = active_choices

    if not choices:
        label = "trained" if requested_view == "auto" else requested_view.upper()
        raise FileNotFoundError(
            f"No {label} model is available. Train models first with "
            "python -m ML.training.train."
        )

    validation_by_model = _full_data_validation_metrics()
    choices = _rank_choices(choices, validation_by_model)
    default_context_path = _context_path()
    context_cache: dict[str, dict] = {}
    vectors: dict[tuple[str, str], np.ndarray] = {}
    feature_errors: dict[str, str] = {}
    model_errors: list[str] = []

    feature_scores: list[dict] = []

    for choice in choices:
        view = choice["feature_set"]
        if view in feature_errors or any(
            result["feature_set"] == view
            for result in feature_scores
        ):
            continue

        try:
            bundle = load(MODEL_DIR / choice["file"])
            selected_context_file = str(
                bundle.get("context_file") or default_context_path.name
            )
            selected_context_path = MODEL_DIR / selected_context_file
            if not selected_context_path.exists():
                raise FileNotFoundError(
                    f"Feature context is missing for {choice['file']}: {selected_context_file}"
                )
            if selected_context_file not in context_cache:
                context_cache[selected_context_file] = load(selected_context_path)
            context = context_cache[selected_context_file]
            cell_line_id = context.get("cell_aliases", {}).get(
                normalize(cell_line), str(cell_line).strip()
            )
            training_cell_line_ids = bundle.get("training_cell_line_ids")
            cell_line_seen_in_training = (
                str(cell_line_id) in set(training_cell_line_ids)
                if training_cell_line_ids is not None
                else None
            )
            vector_key = (view, selected_context_file)
            if vector_key not in vectors:
                vectors[vector_key] = _build_vector(
                    drug_a,
                    drug_b,
                    cell_line,
                    view,
                    context,
                    bundle,
                )
            vector = vectors[vector_key]
            model_for_prediction = bundle["model"]
            selected_model_name = bundle.get("model_name", choice["model"])
            selected_implementation = bundle.get("implementation", "unknown")
            validation = validation_by_model.get((view, choice["model"]))
            if validation is None:
                bundle_metrics = bundle.get("validation_metrics") or {}
                validation = {
                    "rmse": bundle_metrics.get("selection_rmse"),
                    "metric": "Bundle validation RMSE",
                    "folds": bundle_metrics.get("folds_evaluated"),
                    "source": bundle.get("dataset_label"),
                }
            model_selection_reason = None
            model_variant = None
            if (
                view == "structure"
                and bundle.get("structure_routing_enabled")
                and bundle.get("expanded_model") is not None
            ):
                legacy_drugs = set(bundle.get("legacy_structure_drug_keys", []))
                drug_keys = {
                    resolve_drug(drug_a, context),
                    resolve_drug(drug_b, context),
                }
                if drug_keys.issubset(legacy_drugs):
                    model_variant = "structure_legacy_coverage"
                    model_selection_reason = (
                        "The original structure model is retained because both drugs were supported before structure enrichment; it had lower error on matched held-out rows."
                    )
                else:
                    model_for_prediction = bundle["expanded_model"]
                    model_variant = "structure_pubchem_expanded"
                    model_selection_reason = (
                        "The expanded structure model was selected because PubChem enrichment supplied a structure for at least one drug the original model could not represent."
                    )
            elif (
                view == "paper_drs"
                and model_name is None
                and bundle.get("pair_routing_enabled")
                and bundle.get("known_pair_model") is not None
            ):
                pair_key = _paper_pair_key(drug_a, drug_b, context)
                known_pairs = set(bundle.get("known_pair_keys", []))
                if pair_key and pair_key in known_pairs:
                    model_for_prediction = bundle["known_pair_model"]
                    selected_model_name = bundle.get(
                        "known_pair_model_name", "synergyx_attention"
                    )
                    selected_implementation = "pytorch"
                    model_selection_reason = (
                        "SynergyX-inspired attention was selected because this drug pair appears in fallback training data."
                    )
                else:
                    model_selection_reason = (
                        "XGBoost was selected because this drug pair was absent from fallback training; it had slightly lower MSE than attention on pair-held-out validation."
                    )
            score = float(model_for_prediction.predict(vector.reshape(1, -1))[0])
            target_clip = bundle.get("target_clip") or {}
            try:
                clip_low = float(target_clip["lower"])
                clip_high = float(target_clip["upper"])
            except (KeyError, TypeError, ValueError):
                clip_low = clip_high = None
            if (
                clip_low is not None
                and clip_high is not None
                and np.isfinite(clip_low)
                and np.isfinite(clip_high)
                and clip_low <= clip_high
            ):
                score = float(np.clip(score, clip_low, clip_high))
        except (ValueError, KeyError, FileNotFoundError) as exc:
            if requested_view != "auto":
                raise
            feature_errors[view] = str(exc)
            continue
        except Exception as exc:
            if requested_view != "auto":
                raise
            model_errors.append(
                f"{view}/{choice['model']}: {exc}"
            )
            continue

        prediction_result = {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "cell_line": cell_line,
            "feature_set": view,
            "model": selected_model_name,
            "deployment_status": bundle.get("deployment_status", "legacy"),
            "implementation": selected_implementation,
            **({"model_variant": model_variant} if model_variant else {}),
            "predicted_s_score": score,
            "feature_scores": [],
            "synergy_threshold": 10.0,
            "predicted_synergistic": bool(score >= 10.0),
            "target": bundle.get("target") or "DrugComb S score",
            "dataset_label": bundle.get("dataset_label"),
            "training_rows": bundle.get("n_train_rows"),
            "training_cell_lines": bundle.get("n_train_cell_lines"),
            "cell_line_seen_in_training": cell_line_seen_in_training,
            "validation_rmse": validation.get("rmse"),
            "validation_metric": validation.get("metric"),
            "validation_folds": validation.get("folds"),
            "validation_source": validation.get("source"),
            "interpretation": (
                "The >=10 label follows the paper's classification threshold; "
                "this is a research prediction, not clinical guidance."
            ),
        }

        if requested_view != "auto":
            prediction_result["selection_strategy"] = (
                model_selection_reason or "Requested feature view."
            )
            prediction_result["feature_scores"] = [{
                "feature_set": view,
                "model": prediction_result["model"],
                "predicted_s_score": score,
                "target": prediction_result["target"],
                "dataset_label": prediction_result["dataset_label"],
                "training_rows": prediction_result["training_rows"],
                "training_cell_lines": prediction_result["training_cell_lines"],
                "cell_line_seen_in_training": prediction_result["cell_line_seen_in_training"],
                "deployment_status": prediction_result["deployment_status"],
                **({"model_variant": model_variant} if model_variant else {}),
            }]
            for key in ("validation_rmse", "validation_metric", "validation_folds", "validation_source"):
                prediction_result.pop(key, None)
            return prediction_result

        feature_scores.append({
            "feature_set": view,
            "model": prediction_result["model"],
            "implementation": prediction_result["implementation"],
            "prediction_status": "available",
            "predicted_s_score": score,
            "target": prediction_result["target"],
            "dataset_label": prediction_result["dataset_label"],
            "training_rows": prediction_result["training_rows"],
            "training_cell_lines": prediction_result["training_cell_lines"],
            "cell_line_seen_in_training": prediction_result["cell_line_seen_in_training"],
            "validation_rmse": prediction_result["validation_rmse"],
            "validation_metric": prediction_result["validation_metric"],
            "validation_folds": prediction_result["validation_folds"],
            "validation_source": prediction_result["validation_source"],
            "deployment_status": prediction_result["deployment_status"],
            **({"model_variant": model_variant} if model_variant else {}),
            **({"model_selection_reason": model_selection_reason} if model_selection_reason else {}),
        })

    if requested_view == "auto" and feature_scores:
        scores_by_view = {
            result["feature_set"]: result
            for result in feature_scores
        }
        choices_by_view: dict[str, dict[str, str]] = {}
        for choice in choices:
            choices_by_view.setdefault(choice["feature_set"], choice)

        view_order = ("ds", "drs", "structure")
        for view in view_order:
            if view in scores_by_view:
                continue

            choice = choices_by_view.get(view)
            reason = feature_errors.get(view)
            if not reason:
                matching_errors = [
                    error for error in model_errors
                    if error.startswith(f"{view}/")
                ]
                reason = matching_errors[0] if matching_errors else None
            if not reason:
                reason = (
                    "No trained model is available for this view."
                    if choice is None
                    else "A prediction could not be produced for this view."
                )

            scores_by_view[view] = {
                "feature_set": view,
                "model": choice["model"] if choice else None,
                "prediction_status": "unavailable",
                "predicted_s_score": None,
                "reason": reason,
            }

        feature_scores = [scores_by_view[view] for view in view_order]
        # The views are not a calibrated ensemble. Choose the available view
        # with the lowest pair-held-out validation RMSE; when possible, prefer
        # a view whose training data included the entered cell line.
        available_scores = [
            item for item in feature_scores
            if item["prediction_status"] == "available"
        ]
        selected_score = min(
            available_scores,
            key=lambda item: (
                0 if item.get("cell_line_seen_in_training") is True else 1,
                float(item["validation_rmse"])
                if item.get("validation_rmse") is not None
                and np.isfinite(float(item["validation_rmse"]))
                else float("inf"),
                view_order.index(item["feature_set"]),
            ),
        )
        selected_score["is_best"] = True
        for item in feature_scores:
            item.setdefault("is_best", False)
        selected_view = selected_score["feature_set"]
        highest_score = max(
            available_scores,
            key=lambda item: float(item["predicted_s_score"]),
        )
        highest_score["is_highest_score"] = True
        for item in feature_scores:
            item.setdefault("is_highest_score", False)

        highest_predicted_score = float(highest_score["predicted_s_score"])
        # The displayed prediction follows the requested rule: report the
        # largest available S-score across Structure, DS, and DRS. Keep the
        # validation-selected estimate separately for transparency.
        displayed_score = highest_score
        score = highest_predicted_score
        public_feature_scores = [
            {
                key: value
                for key, value in item.items()
                if key not in {"validation_rmse", "validation_metric", "validation_folds", "validation_source"}
            }
            for item in feature_scores
        ]

        return {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "cell_line": cell_line,
            "feature_set": displayed_score["feature_set"],
            "model": displayed_score["model"],
            "implementation": displayed_score["implementation"],
            "deployment_status": displayed_score.get("deployment_status", "candidate"),
            "highest_predicted_s_score": highest_predicted_score,
            "highest_score_feature_set": highest_score["feature_set"],
            "highest_score_model": highest_score["model"],
            "best_validated_feature_set": selected_view,
            "best_validated_model": selected_score["model"],
            "best_validated_s_score": selected_score["predicted_s_score"],
            **({"model_variant": displayed_score["model_variant"]} if displayed_score.get("model_variant") else {}),
            "selection_strategy": (
                f"The displayed score is the largest available estimate across feature views: {highest_predicted_score:.2f} from {highest_score['feature_set'].upper()} ({highest_score['model']}). "
                f"The validation and cell-line-support selection is {selected_view.upper()} ({selected_score['model']}) at {float(selected_score['predicted_s_score']):.2f}; validation rank and score size are reported separately."
            ),
            "predicted_s_score": score,
            "feature_scores": public_feature_scores,
            "synergy_threshold": 10.0,
            "predicted_synergistic": bool(score >= 10.0),
            "target": displayed_score.get("target") or "DrugComb S score",
            "dataset_label": displayed_score.get("dataset_label"),
            "training_rows": displayed_score.get("training_rows"),
            "training_cell_lines": displayed_score.get("training_cell_lines"),
            "cell_line_seen_in_training": displayed_score.get("cell_line_seen_in_training"),
            "interpretation": (
                "The >=10 label follows the paper's classification threshold; "
                "this is a research prediction, not clinical guidance."
            ),
        }

    if requested_view == "auto":
        details = []
        if feature_errors:
            details.extend(
                f"{view}: {message}"
                for view, message in feature_errors.items()
            )
        if model_errors:
            details.extend(model_errors[:2])
        suffix = f" Details: {'; '.join(details)}" if details else ""
        raise ValueError(
            "No trained model could build a prediction for these drugs and "
            f"this cell line.{suffix}"
        )

    raise ValueError(
        f"No usable model is available for the {requested_view.upper()} view."
    )
