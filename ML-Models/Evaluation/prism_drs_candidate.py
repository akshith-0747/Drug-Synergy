"""Evaluate an expanded DRS feature source using official PRISM dose-response data.

This is an isolated candidate experiment. It does not change the prediction
service or production feature context. DrugComb pairs stay grouped across cell
lines in every fold, and both candidates use the same sampled DrugComb rows and
held-out pair assignments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ML.feature_engineering.features import (
    ML_DIR,
    build_context,
    _load_prism_resistance_signatures,
    make_feature_matrix,
)
from ML.training.train import (
    PROCESSED_FILE,
    _group_validation_indices,
    _metrics,
    _model,
    _pair_groups,
    _sample_frame,
    _stratification_labels,
)

OUTPUT_DIR = ML_DIR / "results" / "prism_drs_candidate_pair_cv"
OUTER_SEED = 2027


def _aggregate_predictions(rows: list[dict], model: str, split: str) -> dict:
    selected = [row for row in rows if row["model"] == model and row["split"] == split]
    y_true = np.concatenate([row["y_true"] for row in selected]) if selected else np.asarray([])
    y_pred = np.concatenate([row["y_pred"] for row in selected]) if selected else np.asarray([])
    return {
        "model": model,
        "evaluation_slice": split,
        "rows": int(len(y_true)),
        **(_metrics(y_true, y_pred) if len(y_true) else {}),
    }


def write_model_lab_comparison(output_dir: Path = OUTPUT_DIR) -> Path:
    """Export only this project's paired DRS comparison in the Model Lab schema."""
    source = pd.read_csv(output_dir / "model_comparison.csv")
    labels = {
        "current_coverage": ("synergyx_gdsc_reference", "GDSC-only SynergyX reference", "reference"),
        "expanded_model_on_current_coverage": ("synergyx_prism_fallback", "SynergyX with PRISM fallback", "production"),
        "newly_covered_pairs": ("synergyx_prism_new_coverage", "SynergyX · newly covered rows", "production"),
    }
    metrics = (
        "mse", "rmse", "r2", "pearson", "spearman", "auc_s_ge_10",
        "accuracy_s_ge_10", "balanced_accuracy_s_ge_10", "precision_s_ge_10",
        "recall_s_ge_10", "f1_s_ge_10", "average_precision_s_ge_10",
    )
    expanded = source.loc[
        source["evaluation_slice"].eq("expanded_model_on_current_coverage")
    ]
    epoch_text = (
        f"{int(expanded['recommended_epochs'].iloc[0])} epochs"
        if len(expanded) and pd.notna(expanded["recommended_epochs"].iloc[0])
        else "nested early stop"
    )
    rows = []
    for evaluation_slice, (model_name, _, status) in labels.items():
        selected = source.loc[source["evaluation_slice"].eq(evaluation_slice)]
        for result in selected.to_dict(orient="records"):
            row = {
                "dataset_label": f"DrugComb v1.4 · S · PRISM DRS pair CV · {epoch_text}",
                "target_description": "DrugComb S score",
                "feature_set": "drs",
                "model": model_name,
                "deployment_status": status,
                "split_strategy": "5-fold stratified group CV by unordered drug pair",
                "folds_evaluated": 5,
                "rows_evaluated": int(result["rows"]),
                "unique_pairs_evaluated": None,
                "evaluation_slice": evaluation_slice,
            }
            row.update({metric: result.get(metric) for metric in metrics})
            row["recommended_epochs"] = result.get("recommended_epochs")
            rows.append(row)
    destination = output_dir / "model_lab_comparison.csv"
    pd.DataFrame(rows).to_csv(destination, index=False)
    return destination


def write_repeated_model_lab_comparison(
    output_dirs: tuple[Path, ...],
    destination: Path,
) -> Path:
    """Summarize repeated nested runs without exposing any paper benchmark rows."""
    tables = [pd.read_csv(path / "model_comparison.csv") for path in output_dirs]
    source = pd.concat(tables, ignore_index=True)
    epoch_values = []
    for path in output_dirs:
        folds_file = path / "fold_metrics.csv"
        if not folds_file.exists():
            continue
        folds = pd.read_csv(folds_file)
        if "expanded_selected_epochs" in folds:
            epoch_values.extend(
                pd.to_numeric(folds["expanded_selected_epochs"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )
    recommended_epochs = int(np.ceil(np.median(epoch_values))) if epoch_values else None
    labels = {
        "current_coverage": ("synergyx_gdsc_reference", "reference"),
        "expanded_model_on_current_coverage": ("synergyx_prism_fallback", "production"),
        "newly_covered_pairs": ("synergyx_prism_new_coverage", "production"),
    }
    metrics = (
        "mse", "rmse", "r2", "pearson", "spearman", "auc_s_ge_10",
        "accuracy_s_ge_10", "balanced_accuracy_s_ge_10", "precision_s_ge_10",
        "recall_s_ge_10", "f1_s_ge_10", "average_precision_s_ge_10",
    )
    rows = []
    for evaluation_slice, (model_name, status) in labels.items():
        selected = source.loc[source["evaluation_slice"].eq(evaluation_slice)]
        if selected.empty:
            continue
        row = {
            "dataset_label": "DrugComb v1.4 · S · PRISM DRS nested pair CV · 2 repeats",
            "target_description": "DrugComb S score",
            "feature_set": "drs",
            "model": model_name,
            "deployment_status": status,
            "split_strategy": "2-repeat, 5-fold stratified group CV by unordered drug pair",
            "folds_evaluated": int(5 * len(output_dirs)),
            "rows_evaluated": int(pd.to_numeric(selected["rows"], errors="coerce").sum()),
            "unique_pairs_evaluated": None,
            "evaluation_slice": evaluation_slice,
            "recommended_epochs": recommended_epochs,
        }
        for metric in metrics:
            values = pd.to_numeric(selected[metric], errors="coerce").dropna()
            row[metric] = float(values.mean()) if len(values) else None
        rows.append(row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    return destination


def run(
    max_rows: int = 20000,
    folds: int = 5,
    models: tuple[str, ...] = ("synergyx", "xgboost"),
    seed: int = OUTER_SEED,
    output_dir: Path = OUTPUT_DIR,
    synergyx_epochs: int = 120,
    synergyx_inner_early_stop: bool = False,
) -> dict:
    context = build_context(include_prism_drs=False)
    prism_signatures, prism_audit = _load_prism_resistance_signatures(
        context["aliases"],
        context["expression"],
    )
    current_signatures = context["resistance_signatures"]
    expanded_signatures = dict(current_signatures)
    added_signatures = {
        drug: vector
        for drug, vector in prism_signatures.items()
        if drug not in current_signatures
    }
    expanded_signatures.update(added_signatures)

    raw = pd.read_csv(
        PROCESSED_FILE,
        dtype={"block_id": str, "depmap_id": str},
        low_memory=False,
    )
    raw["S"] = pd.to_numeric(raw["S"], errors="coerce")
    raw = raw.dropna(subset=["S", "depmap_id", "drug_row", "drug_col"])
    raw = _sample_frame(raw, max_rows, seed=42)
    raw["candidate_row_id"] = np.arange(len(raw), dtype=np.int64)

    baseline_X, baseline_y, baseline_rows, baseline_coverage = make_feature_matrix(
        raw, "drs", context
    )
    expanded_context = dict(context)
    expanded_context["resistance_signatures"] = expanded_signatures
    expanded_X, expanded_y, expanded_rows, expanded_coverage = make_feature_matrix(
        raw, "drs", expanded_context
    )
    baseline_groups = _pair_groups(baseline_rows, context)
    expanded_groups = _pair_groups(expanded_rows, expanded_context)

    labels = _stratification_labels(expanded_y, folds=folds)
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    splits = list(splitter.split(expanded_X, labels, expanded_groups))
    all_rows: list[dict] = []
    prediction_records: list[dict] = []
    baseline_ids = baseline_rows["candidate_row_id"].to_numpy(dtype=np.int64)
    expanded_ids = expanded_rows["candidate_row_id"].to_numpy(dtype=np.int64)

    for fold, (_, expanded_test_idx) in enumerate(splits, start=1):
        test_groups = set(expanded_groups[expanded_test_idx])
        base_train_mask = np.asarray([group not in test_groups for group in baseline_groups])
        base_test_mask = np.asarray([group in test_groups for group in baseline_groups])
        expanded_train_mask = np.asarray([group not in test_groups for group in expanded_groups])
        expanded_test_mask = np.asarray([group in test_groups for group in expanded_groups])
        if not base_train_mask.any() or not expanded_train_mask.any():
            raise RuntimeError(f"Fold {fold} has an empty training partition.")

        base_test_indices = np.flatnonzero(base_test_mask)
        expanded_test_indices = np.flatnonzero(expanded_test_mask)
        base_test_ids = set(baseline_ids[base_test_indices].tolist())
        common_mask = np.asarray([row_id in base_test_ids for row_id in expanded_ids[expanded_test_indices]])
        new_test_indices = expanded_test_indices[~common_mask]

        print(
            f"Fold {fold}/{folds}: baseline train={int(base_train_mask.sum()):,}, "
            f"expanded train={int(expanded_train_mask.sum()):,}, "
            f"shared test={len(base_test_indices):,}, newly covered test={len(new_test_indices):,}",
            flush=True,
        )
        for model_name in models:
            model_kwargs = {
                "seed": seed + fold,
                "jobs": 4,
                "drug_dim": int(context["cell_dim"]),
                "cell_dim": int(context["cell_dim"]),
                "synergyx_epochs": int(synergyx_epochs),
            }
            baseline_model = _model(model_name, **model_kwargs)
            base_train_indices = np.flatnonzero(base_train_mask)
            base_selected_epochs = int(synergyx_epochs) if model_name == "synergyx" else None
            if model_name == "synergyx" and synergyx_inner_early_stop:
                inner_train, inner_valid = _group_validation_indices(
                    baseline_groups[base_train_indices],
                    fraction=0.15,
                    seed=seed + fold,
                )
                baseline_model.fit(
                    baseline_X[base_train_indices[inner_train]],
                    baseline_y[base_train_indices[inner_train]],
                    validation_data=(
                        baseline_X[base_train_indices[inner_valid]],
                        baseline_y[base_train_indices[inner_valid]],
                    ),
                )
                base_selected_epochs = int(baseline_model.best_epoch_ or synergyx_epochs)
                baseline_model = _model(
                    model_name,
                    **{**model_kwargs, "synergyx_epochs": base_selected_epochs},
                )
            baseline_model.fit(baseline_X[base_train_indices], baseline_y[base_train_indices])
            baseline_pred = np.asarray(baseline_model.predict(baseline_X[base_test_indices]), dtype=float)
            prediction_records.append({
                "model": model_name,
                "split": "current_coverage",
                "y_true": baseline_y[base_test_indices],
                "y_pred": baseline_pred,
            })

            expanded_model = _model(model_name, **model_kwargs)
            expanded_train_indices = np.flatnonzero(expanded_train_mask)
            expanded_selected_epochs = int(synergyx_epochs) if model_name == "synergyx" else None
            if model_name == "synergyx" and synergyx_inner_early_stop:
                inner_train, inner_valid = _group_validation_indices(
                    expanded_groups[expanded_train_indices],
                    fraction=0.15,
                    seed=seed + fold,
                )
                expanded_model.fit(
                    expanded_X[expanded_train_indices[inner_train]],
                    expanded_y[expanded_train_indices[inner_train]],
                    validation_data=(
                        expanded_X[expanded_train_indices[inner_valid]],
                        expanded_y[expanded_train_indices[inner_valid]],
                    ),
                )
                expanded_selected_epochs = int(
                    expanded_model.best_epoch_ or synergyx_epochs
                )
                expanded_model = _model(
                    model_name,
                    **{**model_kwargs, "synergyx_epochs": expanded_selected_epochs},
                )
            expanded_model.fit(expanded_X[expanded_train_indices], expanded_y[expanded_train_indices])
            expanded_common_idx = expanded_test_indices[common_mask]
            expanded_common_pred = np.asarray(
                expanded_model.predict(expanded_X[expanded_common_idx]), dtype=float
            )
            expanded_new_pred = np.asarray(
                expanded_model.predict(expanded_X[new_test_indices]), dtype=float
            ) if len(new_test_indices) else np.asarray([], dtype=float)
            prediction_records.extend([
                {
                    "model": model_name,
                    "split": "expanded_model_on_current_coverage",
                    "y_true": expanded_y[expanded_common_idx],
                    "y_pred": expanded_common_pred,
                },
                {
                    "model": model_name,
                    "split": "newly_covered_pairs",
                    "y_true": expanded_y[new_test_indices],
                    "y_pred": expanded_new_pred,
                },
            ])
            all_rows.append({
                "fold": fold,
                "model": model_name,
                "current_test_rows": int(len(base_test_indices)),
                "newly_covered_test_rows": int(len(new_test_indices)),
                "test_pairs": int(len(test_groups)),
                "current_training_rows": int(base_train_mask.sum()),
                "expanded_training_rows": int(expanded_train_mask.sum()),
                "current_selected_epochs": base_selected_epochs,
                "expanded_selected_epochs": expanded_selected_epochs,
                "current_metrics": _metrics(baseline_y[base_test_indices], baseline_pred),
                "expanded_current_metrics": _metrics(expanded_y[expanded_common_idx], expanded_common_pred),
                "new_coverage_metrics": _metrics(expanded_y[new_test_indices], expanded_new_pred) if len(new_test_indices) else {},
            })

    aggregate = []
    for model_name in models:
        for split in (
            "current_coverage",
            "expanded_model_on_current_coverage",
            "newly_covered_pairs",
        ):
            aggregate.append(_aggregate_predictions(prediction_records, model_name, split))
    for row in aggregate:
        if row["model"] != "synergyx":
            continue
        epoch_column = (
            "current_selected_epochs"
            if row["evaluation_slice"] == "current_coverage"
            else "expanded_selected_epochs"
        )
        epochs = [
            int(result[epoch_column])
            for result in all_rows
            if result["model"] == "synergyx"
            and result[epoch_column] is not None
        ]
        if epochs:
            row["recommended_epochs"] = int(np.ceil(np.median(epochs)))

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(aggregate).to_csv(output_dir / "model_comparison.csv", index=False)
    write_model_lab_comparison(output_dir)
    pd.DataFrame([
        {
            key: value
            for key, value in record.items()
            if key not in {"current_metrics", "expanded_current_metrics", "new_coverage_metrics"}
        }
        | {f"current_{key}": value for key, value in record["current_metrics"].items()}
        | {f"expanded_{key}": value for key, value in record["expanded_current_metrics"].items()}
        | {f"new_{key}": value for key, value in record["new_coverage_metrics"].items()}
        for record in all_rows
    ]).to_csv(output_dir / "fold_metrics.csv", index=False)

    report = {
        "title": "PRISM-backed DRS fallback candidate",
        "target_dataset": "DrugComb v1.4 S score",
        "validation": {
            "rows_sampled": int(len(raw)),
            "folds": int(folds),
            "grouping_key": "unordered drug pair across all cell lines",
            "same_test_groups_for_both_candidates": True,
            "target_clipping": False,
            "seed": int(seed),
            "synergyx_epochs": int(synergyx_epochs),
            "synergyx_inner_early_stop": bool(synergyx_inner_early_stop),
            "models": list(models),
        },
        "baseline_coverage": baseline_coverage,
        "expanded_coverage": expanded_coverage,
        "prism_source_audit": prism_audit,
        "new_prism_signature_drugs": int(len(added_signatures)),
        "metrics": aggregate,
        "fold_metrics_file": "fold_metrics.csv",
        "model_comparison_file": "model_comparison.csv",
        "promotion_status": "candidate only; production is not modified by this script",
    }
    (output_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# PRISM-backed DRS candidate\n\n"
        "This experiment tests a PRISM fallback for drugs with no current GDSC-derived DRS signature. "
        "It does not overwrite or retrain production models. PRISM secondary-screen dose-response curves "
        "are filtered to STR-passing cell lines, R² >= 0.5, and positive IC50, with MTS010 preferred "
        "when a quality-passing curve exists. Response is log10(IC50 in micromolar); each signature is "
        "resistant-minus-sensitive expression using at least 10 matched cell lines.\n\n"
        "Both models use the same target sample and the same held-out unordered drug-pair folds. "
        "The comparison reports performance on currently covered rows and the expanded model's results "
        "on newly covered rows separately. See `run_report.json`, `fold_metrics.csv`, and `model_comparison.csv`.\n\n"
        "Source: DepMap PRISM Repurposing secondary-screen data, https://depmap.org/repurposing/. "
        "The response assay is a distinct monotherapy viability assay; more feature coverage alone does not "
        "establish better accuracy. Promote only if held-out metrics support it.\n",
        encoding="utf-8",
    )
    print("\nAggregate results:")
    print(pd.DataFrame(aggregate).to_string(index=False))
    print(f"\nSaved candidate report to {output_dir}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=20000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--models", default="synergyx,xgboost")
    parser.add_argument("--seed", type=int, default=OUTER_SEED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--synergyx-epochs", type=int, default=120)
    parser.add_argument("--synergyx-inner-early-stop", action="store_true")
    args = parser.parse_args()
    run(
        max_rows=args.max_rows,
        folds=args.folds,
        models=tuple(value.strip() for value in args.models.split(",") if value.strip()),
        seed=args.seed,
        output_dir=args.output_dir,
        synergyx_epochs=args.synergyx_epochs,
        synergyx_inner_early_stop=args.synergyx_inner_early_stop,
    )
