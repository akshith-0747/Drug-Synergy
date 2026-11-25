"""Compare old and PubChem-expanded structure models on matched pair holdouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ML.feature_engineering.features import ML_DIR, make_feature_matrix
from ML.storage import load
from ML.training.train import _metrics, _model, _pair_groups, _stratification_labels


RESULTS_DIR = ML_DIR / "results" / "pubchem_enrichment_evaluation"
PROCESSED_DIR = ML_DIR / "data" / "processed"
OLD_DATA_FILE = PROCESSED_DIR / "matched_drugcomb_pre_pubchem_20261123.csv"
NEW_DATA_FILE = PROCESSED_DIR / "matched_drugcomb.csv"
OLD_CONTEXT_FILE = ML_DIR / "results" / "models" / "feature_context_pre_pubchem_20261123.joblib"
NEW_CONTEXT_FILE = RESULTS_DIR / "models" / "feature_context.joblib"


def run(seeds: list[int]) -> dict:
    old = pd.read_csv(OLD_DATA_FILE, dtype={"block_id": str, "depmap_id": str}, low_memory=False)
    old["S"] = pd.to_numeric(old["S"], errors="coerce")
    old = old.dropna(subset=["S", "depmap_id", "drug_row", "drug_col"]).reset_index(drop=True)
    new = pd.read_csv(NEW_DATA_FILE, dtype={"block_id": str, "depmap_id": str}, low_memory=False)
    new["S"] = pd.to_numeric(new["S"], errors="coerce")
    new = new.dropna(subset=["S", "depmap_id", "drug_row", "drug_col"]).reset_index(drop=True)
    shared_columns = ["block_id", "drug_row", "drug_col", "cell_line_name", "depmap_id", "S"]
    if len(old) != len(new) or not np.array_equal(
        old[shared_columns].fillna("").astype(str).to_numpy(),
        new[shared_columns].fillna("").astype(str).to_numpy(),
    ):
        raise RuntimeError("The old and enriched target-valid rows do not align exactly.")

    old["_paired_row_id"] = np.arange(len(old), dtype=np.int64)
    new["_paired_row_id"] = np.arange(len(new), dtype=np.int64)
    old_context = load(OLD_CONTEXT_FILE)
    new_context = load(NEW_CONTEXT_FILE)
    X_old, y_old, old_rows, old_coverage = make_feature_matrix(old, "structure", old_context)
    X_new, y_new, new_rows, new_coverage = make_feature_matrix(new, "structure", new_context)
    old_ids = old_rows["_paired_row_id"].to_numpy(dtype=np.int64)
    new_ids = new_rows["_paired_row_id"].to_numpy(dtype=np.int64)
    old_pos = {int(row_id): pos for pos, row_id in enumerate(old_ids)}
    new_pos = {int(row_id): pos for pos, row_id in enumerate(new_ids)}
    groups = _pair_groups(new_rows, new_context)
    labels = _stratification_labels(y_new, folds=5)
    records = []

    for seed in seeds:
        split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        train_idx, test_idx = next(split.split(X_new, labels, groups))
        train_ids = set(new_ids[train_idx].tolist())
        test_ids = new_ids[test_idx].tolist()
        old_train_pos = np.asarray(
            [old_pos[row_id] for row_id in train_ids if row_id in old_pos],
            dtype=np.int64,
        )
        common_ids = [row_id for row_id in test_ids if row_id in old_pos]
        recovered_ids = [row_id for row_id in test_ids if row_id not in old_pos]
        old_test_pos = np.asarray([old_pos[row_id] for row_id in common_ids], dtype=np.int64)
        new_common_pos = np.asarray([new_pos[row_id] for row_id in common_ids], dtype=np.int64)
        new_recovered_pos = np.asarray([new_pos[row_id] for row_id in recovered_ids], dtype=np.int64)

        if set(groups[train_idx]).intersection(set(groups[test_idx])):
            raise RuntimeError("An unordered drug pair appears in both train and test.")

        old_model = _model("xgboost", seed, -1)
        old_model.fit(X_old[old_train_pos], y_old[old_train_pos])
        expanded_model = _model("xgboost", seed, -1)
        expanded_model.fit(X_new[train_idx], y_new[train_idx])

        old_common_pred = old_model.predict(X_old[old_test_pos])
        expanded_common_pred = expanded_model.predict(X_new[new_common_pos])
        expanded_recovered_pred = expanded_model.predict(X_new[new_recovered_pos])
        expanded_all_pred = expanded_model.predict(X_new[test_idx])

        routed_pred = np.empty(len(test_ids), dtype=np.float64)
        old_test_local = {row_id: pos for pos, row_id in enumerate(common_ids)}
        new_recovered_local = {row_id: pos for pos, row_id in enumerate(recovered_ids)}
        for local_pos, row_id in enumerate(test_ids):
            if row_id in old_test_local:
                routed_pred[local_pos] = old_common_pred[old_test_local[row_id]]
            else:
                routed_pred[local_pos] = expanded_recovered_pred[new_recovered_local[row_id]]

        records.extend([
            {
                "seed": seed,
                "evaluation_group": "old_supported_test_rows",
                "model": "legacy_structure_xgboost",
                "test_rows": len(common_ids),
                "test_pairs": len(set(groups[test_idx][np.isin(new_ids[test_idx], common_ids)].tolist())),
                **_metrics(y_new[new_common_pos], old_common_pred),
            },
            {
                "seed": seed,
                "evaluation_group": "old_supported_test_rows",
                "model": "expanded_structure_xgboost",
                "test_rows": len(common_ids),
                "test_pairs": len(set(groups[test_idx][np.isin(new_ids[test_idx], common_ids)].tolist())),
                **_metrics(y_new[new_common_pos], expanded_common_pred),
            },
            {
                "seed": seed,
                "evaluation_group": "newly_recovered_test_rows",
                "model": "expanded_structure_xgboost",
                "test_rows": len(recovered_ids),
                "test_pairs": len(set(groups[test_idx][np.isin(new_ids[test_idx], recovered_ids)].tolist())),
                **_metrics(y_new[new_recovered_pos], expanded_recovered_pred),
            },
            {
                "seed": seed,
                "evaluation_group": "all_test_rows",
                "model": "expanded_structure_xgboost",
                "test_rows": len(test_ids),
                "test_pairs": len(set(groups[test_idx].tolist())),
                **_metrics(y_new[test_idx], expanded_all_pred),
            },
            {
                "seed": seed,
                "evaluation_group": "all_test_rows",
                "model": "pair_routed_structure_xgboost",
                "test_rows": len(test_ids),
                "test_pairs": len(set(groups[test_idx].tolist())),
                **_metrics(y_new[test_idx], routed_pred),
            },
        ])
        print(
            f"seed={seed} shared_rows={len(common_ids):,} recovered_rows={len(recovered_ids):,} "
            f"legacy_MSE={records[-5]['mse']:.3f} expanded_shared_MSE={records[-4]['mse']:.3f} "
            f"routed_MSE={records[-1]['mse']:.3f} expanded_all_MSE={records[-2]['mse']:.3f}",
            flush=True,
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(records)
    result_path = RESULTS_DIR / "pubchem_structure_route_holdout.csv"
    result_frame.to_csv(result_path, index=False)
    grouped = (
        result_frame.groupby(["evaluation_group", "model"], as_index=False)
        .agg(seeds=("seed", "nunique"), rows_evaluated=("test_rows", "mean"),
             pairs_evaluated=("test_pairs", "mean"), mse=("mse", "mean"),
             rmse=("rmse", "mean"), r2=("r2", "mean"))
    )
    report = {
        "protocol": "One stratified group fold per seed; all rows for an unordered drug pair are held out together.",
        "seeds": seeds,
        "old_coverage": old_coverage,
        "expanded_coverage": new_coverage,
        "legacy_drug_count": len(old_context.get("fingerprints", {})),
        "expanded_drug_count": len(new_context.get("fingerprints", {})),
        "fold_metrics_file": str(result_path.relative_to(ML_DIR.parent)),
        "summary": grouped.to_dict(orient="records"),
        "deployment_rule": "Use the legacy structure model when both drugs were supported by its training context; use the expanded model when at least one newly resolved structure is required.",
    }
    report_path = RESULTS_DIR / "pubchem_structure_route_holdout.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(grouped.to_string(index=False), flush=True)
    print(f"Saved {report_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="2028,2029,2030")
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        parser.error("At least one seed is required.")
    run(seeds)


if __name__ == "__main__":
    main()
