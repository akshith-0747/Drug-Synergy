"""Compare structure models on DrugComb pairs absent from the training sample.

This check uses the same S-score-stratified 20,000-row sample as the saved
structure CV run, then tests only unordered drug pairs not present in that
sample. The test rows come from the same DrugComb source, so this is a held-out
pair check, not independent-study validation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ML.feature_engineering.features import (
    ML_DIR,
    augment_structure_features,
    build_context,
    make_feature_matrix,
    resolve_drug,
)
from ML.storage import load
from ML.training.train import PROCESSED_FILE, _metrics, _sample_frame

RESULTS_DIR = ML_DIR / "results"
MODEL_DIR = RESULTS_DIR / "models"
TRAINING_SAMPLE_ROWS = 20_000


def _unordered_pair_keys(frame: pd.DataFrame, context: dict) -> np.ndarray:
    """Build canonical, order-independent pair keys with one lookup per drug."""
    values = frame[["drug_row", "drug_col"]].to_numpy(dtype=object)
    unique_drugs = pd.unique(values.reshape(-1))
    resolved = {drug: resolve_drug(drug, context) for drug in unique_drugs}
    left = np.fromiter(
        (resolved[value] for value in values[:, 0]),
        dtype=object,
        count=len(values),
    )
    right = np.fromiter(
        (resolved[value] for value in values[:, 1]),
        dtype=object,
        count=len(values),
    )
    first, second = np.where(left <= right, left, right), np.where(left <= right, right, left)
    return np.char.add(np.char.add(first.astype(str), "\x1f"), second.astype(str))


def run() -> dict:
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

    training_sample = _sample_frame(raw, TRAINING_SAMPLE_ROWS, seed=42)
    _, y_train, used_train, train_coverage = make_feature_matrix(
        training_sample, "structure", context
    )
    training_pairs = set(_unordered_pair_keys(used_train, context))

    all_pair_keys = _unordered_pair_keys(raw, context)
    unseen_pair_mask = ~np.isin(all_pair_keys, list(training_pairs))
    heldout_frame = raw.loc[unseen_pair_mask].reset_index(drop=True)
    X_test, y_test, used_test, test_coverage = make_feature_matrix(
        heldout_frame, "structure", context
    )
    heldout_pairs = set(_unordered_pair_keys(used_test, context))
    if not heldout_pairs or not training_pairs.isdisjoint(heldout_pairs):
        raise RuntimeError("The held-out set contains a training drug pair.")

    rows = []
    schemas = {}
    for name in ("xgboost", "xgboost_pairwise"):
        bundle = load(MODEL_DIR / f"structure__{name}.joblib")
        model = bundle["model"]
        predicted = np.asarray(model.predict(X_test), dtype=float)
        schema = bundle.get("pair_schema", "concat_v1")
        schemas[name] = schema
        rows.append(
            {
                "feature_set": "structure",
                "model": name,
                "pair_schema": schema,
                "split_strategy": "held_out_unordered_drug_pairs_from_full_drugcomb",
                "training_rows": int(len(y_train)),
                "training_pairs": int(len(training_pairs)),
                "test_rows": int(len(y_test)),
                "test_pairs": int(len(heldout_pairs)),
                **_metrics(y_test, predicted),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(RESULTS_DIR / "structure_pair_holdout.csv", index=False)
    report = {
        "target": "DrugComb S score",
        "training_sample_requested_rows": TRAINING_SAMPLE_ROWS,
        "training_feature_coverage": train_coverage,
        "heldout_feature_coverage": test_coverage,
        "training_rows": int(len(y_train)),
        "training_pairs": int(len(training_pairs)),
        "heldout_rows": int(len(y_test)),
        "heldout_pairs": int(len(heldout_pairs)),
        "pair_overlap": 0,
        "model_schemas": schemas,
        "validation_scope": "Held-out unordered drug pairs drawn from the remaining rows of the same DrugComb source; not external-study validation.",
        "metrics_file": "ML/results/structure_pair_holdout.csv",
    }
    (RESULTS_DIR / "structure_pair_holdout_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(result.to_string(index=False))
    print(
        f"\nHeld out {len(heldout_pairs):,} drug pairs and {len(y_test):,} rows; "
        "pair overlap with training: 0."
    )
    return report


if __name__ == "__main__":
    run()
