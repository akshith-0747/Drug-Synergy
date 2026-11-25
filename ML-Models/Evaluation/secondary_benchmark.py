"""Secondary held-out-key benchmark against the supplied ZIP-score table.

This is not an independent-study validation: the scored table overlaps some
DrugComb source records, so exact drug-pair/cell-line keys seen in training
are removed and the score mismatch is stated in the report.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ML.feature_engineering.features import DATA_DIR, ML_DIR, normalize, pair_vector
from ML.models.native_fallback import auc_score
from ML.prediction.predict import available_models
from ML.storage import load

SCORED_FILE = DATA_DIR / "drugcombs_scored.csv.xls"
PROCESSED_FILE = DATA_DIR / "processed" / "matched_drugcomb.csv"
MODEL_DIR = ML_DIR / "results" / "models"
RESULTS_DIR = ML_DIR / "results"


def _key(drug_a, drug_b, cell):
    a, b = sorted((normalize(drug_a), normalize(drug_b)))
    return f"{a}::{b}::{normalize(cell)}"


def _correlation(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else float("nan")


def _average_precision(y_true, scores):
    y_true = np.asarray(y_true, dtype=np.int8)
    if not y_true.any():
        return float("nan")
    order = np.argsort(np.asarray(scores))[::-1]
    positives = y_true[order]
    precision = np.cumsum(positives) / (np.arange(len(positives)) + 1)
    return float(np.sum(precision * positives) / positives.sum())


def _context_path():
    for suffix in (".joblib", ".pkl"):
        candidate = MODEL_DIR / f"feature_context{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Train models first so the feature context is saved.")


def main(max_rows: int = 50000):
    if not SCORED_FILE.exists() or not PROCESSED_FILE.exists():
        raise FileNotFoundError("The scored ZIP table and prepared DrugComb file are required.")
    train = pd.read_csv(PROCESSED_FILE, usecols=["drug_row", "drug_col", "cell_line_name"], dtype=str, encoding_errors="replace")
    seen = set(_key(r.drug_row, r.drug_col, r.cell_line_name) for r in train.itertuples(index=False))
    frame = pd.read_csv(SCORED_FILE, dtype=str, encoding="utf-8-sig", encoding_errors="replace")
    required = {"Drug1", "Drug2", "Cell line", "ZIP"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Scored table is missing columns: {sorted(missing)}")
    frame["ZIP"] = pd.to_numeric(frame["ZIP"], errors="coerce")
    frame = frame.dropna(subset=["ZIP"])
    frame["_key"] = [_key(a, b, c) for a, b, c in zip(frame["Drug1"], frame["Drug2"], frame["Cell line"])]
    frame = frame[~frame["_key"].isin(seen)].reset_index(drop=True)
    if max_rows > 0 and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=42).reset_index(drop=True)
    context = load(_context_path())
    output = []
    for item in available_models():
        bundle = load(MODEL_DIR / item["file"])
        vectors, targets = [], []
        # Select columns explicitly: the source table's column order can vary.
        for drug_a, drug_b, cell_line, zip_score in frame[["Drug1", "Drug2", "Cell line", "ZIP"]].itertuples(index=False, name=None):
            try:
                vector = pair_vector(
                    drug_a,
                    drug_b,
                    cell_line,
                    item["feature_set"],
                    context,
                    pair_schema=bundle.get("pair_schema", "concat_v1"),
                )
            except (ValueError, AttributeError):
                continue
            vectors.append(vector)
            targets.append(float(zip_score))
        if not vectors:
            continue
        y_zip = np.asarray(targets, dtype=float)
        y_pred_s = np.asarray(bundle["model"].predict(np.vstack(vectors)), dtype=float)
        observed_synergy = (y_zip >= 10).astype(np.int8)
        output.append({
            "feature_set": item["feature_set"], "model": item["model"],
            "implementation": bundle.get("implementation", "unknown"),
            "rows_predicted": len(y_zip), "zip_positive_rate_ge_10": float(np.mean(observed_synergy)),
            "auc_zip_ge_10": auc_score(observed_synergy, y_pred_s),
            "average_precision_zip_ge_10": _average_precision(observed_synergy, y_pred_s),
            "spearman_zip_vs_predicted_s": float(pd.Series(y_zip).corr(pd.Series(y_pred_s).rank(), method="pearson")),
            "pearson_zip_vs_predicted_s": _correlation(y_zip, y_pred_s),
            "warning": "Secondary ZIP-score benchmark after removing exact pair-cell keys seen in training; not independent-study validation and ZIP is not the S-score target.",
        })
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_file = RESULTS_DIR / "secondary_zip_benchmark.csv"
    pd.DataFrame(output).to_csv(result_file, index=False)
    print(f"Remaining scored rows after exact-key exclusion: {len(frame):,}")
    print(f"Saved secondary benchmark: {result_file}")


if __name__ == "__main__":
    main()

