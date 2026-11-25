"""Compare author-released DRS/multi-omics features with the current DRS view.

This is a research benchmark, not a production training command. It evaluates
both the paper-style random row split and the stricter unseen-pair split. The
author-released feature arrays are read from ML/data/processed/paper_reference.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, StratifiedGroupKFold, StratifiedShuffleSplit

from ML.feature_engineering.features import ML_DIR, build_context, make_feature_matrix
from ML.feature_engineering.paper_release import load_author_feature_maps
from ML.training.train import _metrics, _model, _stratification_labels


RESULTS_DIR = ML_DIR / "results" / "paper_feature_benchmark"
DATA_FILE = ML_DIR / "data" / "processed" / "matched_drugcomb.csv"
def _canonical_smiles(value: object) -> str | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True)


def _load_author_features() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    return load_author_feature_maps()


def _build_candidate_rows() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict]:
    raw = pd.read_csv(
        DATA_FILE,
        dtype={"block_id": str, "depmap_id": str},
        low_memory=False,
    )
    raw["S"] = pd.to_numeric(raw["S"], errors="coerce")
    raw = raw.dropna(subset=["S", "depmap_id", "drug_row", "drug_col"]).reset_index(drop=True)
    raw["_benchmark_row_id"] = np.arange(len(raw), dtype=np.int64)

    canonical_cache: dict[str, str | None] = {}
    for column in ("smiles_row", "smiles_col"):
        if column not in raw:
            raw[column] = ""
        values = raw[column].fillna("").astype(str)
        for value in values.unique():
            canonical_cache[value] = _canonical_smiles(value)
        raw[f"_{column}"] = values.map(canonical_cache)

    drug_features, cell_features = _load_author_features()
    keep = (
        raw["_smiles_row"].isin(drug_features)
        & raw["_smiles_col"].isin(drug_features)
        & raw["depmap_id"].astype(str).isin(cell_features)
    )
    selected = raw.loc[keep].copy().reset_index(drop=True)
    if selected.empty:
        raise RuntimeError("No DrugComb rows match the author-released DRS and cell features.")

    drug_order = sorted(drug_features)
    cell_order = sorted(cell_features)
    drug_index = {smiles: i for i, smiles in enumerate(drug_order)}
    cell_index = {cell_id: i for i, cell_id in enumerate(cell_order)}
    drug_matrix = np.stack([drug_features[smiles] for smiles in drug_order])
    cell_matrix = np.stack([cell_features[cell_id] for cell_id in cell_order])

    drug_a = selected["_smiles_row"].map(drug_index).to_numpy(dtype=np.int64)
    drug_b = selected["_smiles_col"].map(drug_index).to_numpy(dtype=np.int64)
    cell_idx = selected["depmap_id"].astype(str).map(cell_index).to_numpy(dtype=np.int64)
    X = np.concatenate(
        [drug_matrix[drug_a], drug_matrix[drug_b], cell_matrix[cell_idx]],
        axis=1,
    ).astype(np.float32, copy=False)
    y = selected["S"].to_numpy(dtype=np.float32)
    pairs = np.asarray(
        ["|".join(sorted(pair)) for pair in zip(selected["_smiles_row"], selected["_smiles_col"])],
        dtype=object,
    )
    drug_dim = int(drug_matrix.shape[1])
    cell_dim = int(cell_matrix.shape[1])
    metadata = {
        "input_rows": int(len(raw)),
        "feature_rows": int(len(selected)),
        "coverage_fraction": float(len(selected) / max(len(raw), 1)),
        "unique_drugs": int(len(drug_order)),
        "unique_pairs": int(len(set(pairs.tolist()))),
        "cell_lines": int(len(set(selected["depmap_id"].astype(str)))),
        "drug_feature_dim": drug_dim,
        "cell_feature_dim": cell_dim,
        "total_feature_dim": int(X.shape[1]),
    }
    return selected, X, y, pairs, metadata


def _build_project_view_rows(
    frame: pd.DataFrame,
    X_paper: np.ndarray,
    y: np.ndarray,
    feature_set: str,
):
    context = build_context()
    X_project, y_project, used_rows, coverage = make_feature_matrix(
        frame.copy(), feature_set, context
    )
    row_ids = used_rows["_benchmark_row_id"].to_numpy(dtype=np.int64)
    current_frame = frame.set_index("_benchmark_row_id")
    smiles_a = current_frame.loc[row_ids, "_smiles_row"].to_numpy(dtype=object)
    smiles_b = current_frame.loc[row_ids, "_smiles_col"].to_numpy(dtype=object)
    current_groups = np.asarray(
        ["|".join(sorted(pair)) for pair in zip(smiles_a, smiles_b)],
        dtype=object,
    )
    # make_feature_matrix resets used rows; use the retained row ID to recover
    # the exact same examples in the paper-release feature matrix.
    candidate_lookup = {
        int(row_id): position
        for position, row_id in enumerate(frame["_benchmark_row_id"].to_numpy(dtype=np.int64))
    }
    paper_positions = np.asarray([candidate_lookup[int(row_id)] for row_id in row_ids])
    if not np.allclose(y_project, y[paper_positions]):
        raise RuntimeError("Project and paper feature rows do not align on target S scores.")
    _, _, drs_rows, _ = make_feature_matrix(frame.copy(), "drs", context)
    drs_row_ids = set(drs_rows["_benchmark_row_id"].astype(np.int64).tolist())
    drs_presence = frame["_benchmark_row_id"].astype(np.int64).isin(drs_row_ids).to_numpy()
    return (
        X_project,
        X_paper[paper_positions],
        y_project,
        current_groups,
        coverage,
        int(context["cell_dim"]),
        drs_presence[paper_positions],
    )


def _inner_split(train_idx: np.ndarray, y: np.ndarray, groups: np.ndarray, protocol: str, seed: int):
    if protocol == "pair_grouped":
        split = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        inner_train, inner_valid = next(
            split.split(np.zeros(len(train_idx)), y[train_idx], groups[train_idx])
        )
    else:
        labels = _stratification_labels(y[train_idx], folds=5)
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        inner_train, inner_valid = next(split.split(np.zeros(len(train_idx)), labels))
    return train_idx[inner_train], train_idx[inner_valid]


def _evaluate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    protocol: str,
    feature_source: str,
    cell_dim: int,
    drs_presence: np.ndarray | None,
    models: list[str],
    repeats: int,
    epochs: int,
) -> pd.DataFrame:
    output = []
    labels = _stratification_labels(y, folds=5)
    for repeat in range(repeats):
        seed = 2027 + repeat
        if protocol == "pair_grouped":
            splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            splits = splitter.split(X, labels, groups)
        else:
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            splits = splitter.split(X, labels)
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            for model_name in models:
                model_seed = seed + fold
                if model_name == "synergyx":
                    inner_train, inner_valid = _inner_split(
                        train_idx, y, groups, protocol, model_seed
                    )
                    tuner = _model(
                        "synergyx", model_seed, -1,
                        drug_dim=(X.shape[1] - cell_dim) // 2,
                        cell_dim=cell_dim,
                        synergyx_epochs=epochs,
                    )
                    tuner.fit(
                        X[inner_train], y[inner_train],
                        validation_data=(X[inner_valid], y[inner_valid]),
                    )
                    selected_epochs = int(tuner.best_epoch_ or epochs)
                    model = _model(
                        "synergyx", model_seed, -1,
                        drug_dim=(X.shape[1] - cell_dim) // 2,
                        cell_dim=cell_dim,
                        synergyx_epochs=selected_epochs,
                    )
                    model.fit(X[train_idx], y[train_idx])
                elif model_name == "xgboost":
                    model = _model("xgboost", model_seed, -1)
                    model.fit(X[train_idx], y[train_idx])
                    selected_epochs = None
                else:
                    raise ValueError(f"Unsupported benchmark model: {model_name}")
                prediction = np.asarray(model.predict(X[test_idx]), dtype=float)
                group_rows = [("all", np.ones(len(test_idx), dtype=bool))]
                if drs_presence is not None:
                    group_rows.extend(
                        [
                            ("current_drs_available", drs_presence[test_idx]),
                            ("current_drs_missing", ~drs_presence[test_idx]),
                        ]
                    )
                for coverage_group, local_mask in group_rows:
                    if not local_mask.any():
                        continue
                    selected_idx = test_idx[local_mask]
                    output.append(
                        {
                            "protocol": protocol,
                            "feature_source": feature_source,
                            "coverage_group": coverage_group,
                            "model": model_name,
                            "repeat": repeat + 1,
                            "fold": fold,
                            "test_rows": int(len(selected_idx)),
                            "test_pairs": int(len(set(groups[selected_idx].tolist()))),
                            "selected_epochs": selected_epochs,
                            **_metrics(y[selected_idx], prediction[local_mask]),
                        }
                    )
                print(
                    f"{protocol} repeat={repeat + 1} fold={fold} {model_name}: "
                    f"MSE={_metrics(y[test_idx], prediction)['mse']:.3f}, "
                    f"RMSE={_metrics(y[test_idx], prediction)['rmse']:.3f}, "
                    f"R2={_metrics(y[test_idx], prediction)['r2']:.3f}",
                    flush=True,
                )
    return pd.DataFrame(output)


def run(protocol: str, models: list[str], repeats: int, epochs: int, project_view: str) -> dict:
    started = time.time()
    frame, X_paper, y_paper, groups_paper, metadata = _build_candidate_rows()
    if X_paper.shape[1] < 66:
        raise RuntimeError(f"Unexpected candidate feature width: {X_paper.shape[1]}.")
    print(f"Paper-release feature rows: {metadata}", flush=True)

    X_project, X_paper_shared, y_shared, groups_shared, project_coverage, project_cell_dim, drs_presence = _build_project_view_rows(
        frame, X_paper, y_paper, project_view
    )
    print(
        f"Project {project_view} shared rows: {project_coverage['usable_rows']:,}; "
        f"project input columns={X_project.shape[1]}, paper input columns={X_paper_shared.shape[1]}",
        flush=True,
    )

    # Evaluate both feature sources on exactly the same rows and folds.
    project_metrics = _evaluate(
        X_project, y_shared, groups_shared, protocol, f"project_{project_view}",
        project_cell_dim, drs_presence,
        models, repeats, epochs,
    )
    paper_metrics = _evaluate(
        X_paper_shared, y_shared, groups_shared, protocol, "paper_release_features",
        int(metadata["cell_feature_dim"]), drs_presence, models, repeats, epochs,
    )
    all_metrics = pd.concat([project_metrics, paper_metrics], ignore_index=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_tag = "_".join(models)
    run_tag = f"{protocol}_{project_view}_{model_tag}_r{repeats}"
    fold_path = RESULTS_DIR / f"{run_tag}_fold_metrics.csv"
    all_metrics.to_csv(fold_path, index=False)
    summary = (
        all_metrics.groupby(
            ["protocol", "feature_source", "coverage_group", "model"],
            as_index=False,
        )
        .agg(
            folds=("fold", "count"),
            rows=("test_rows", "sum"),
            pairs=("test_pairs", "sum"),
            mse=("mse", "mean"),
            mse_sd=("mse", "std"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
            pearson=("pearson", "mean"),
            spearman=("spearman", "mean"),
        )
    )
    summary_path = RESULTS_DIR / f"{run_tag}_summary.csv"
    summary.to_csv(summary_path, index=False)
    report = {
        "protocol": protocol,
        "models": models,
        "repeats": repeats,
        "project_view": project_view,
        "coverage": metadata,
        "project_view_shared_row_coverage": project_coverage,
        "paper_feature_benchmark_is_not_production": True,
        "paper_internal_mse_reference": 92.16,
        "note": "Feature-source comparison uses identical rows and outer folds. Pair-grouped results estimate unseen drug-pair performance; paper-style random row results are for protocol comparison only. The author feature files are not the complete paper model implementation.",
        "fold_metrics_file": str(fold_path.relative_to(ML_DIR.parent)),
        "summary_file": str(summary_path.relative_to(ML_DIR.parent)),
        "elapsed_seconds": time.time() - started,
        "summary": summary.to_dict(orient="records"),
    }
    report_path = RESULTS_DIR / f"{run_tag}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=["paper", "pair_grouped"],
        default="pair_grouped",
        help="paper: stratified random rows, for comparison with the PDF; pair_grouped: unseen unordered drug pairs.",
    )
    parser.add_argument(
        "--project-view",
        choices=["structure", "ds", "drs"],
        default="drs",
        help="Existing project feature view to compare on the same available rows.",
    )
    parser.add_argument("--models", default="xgboost,synergyx")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    models = [name.strip().lower() for name in args.models.split(",") if name.strip()]
    run(args.protocol, models, args.repeats, args.epochs, args.project_view)


if __name__ == "__main__":
    main()
