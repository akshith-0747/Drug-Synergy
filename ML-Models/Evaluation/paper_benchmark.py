"""Evaluate saved models on the benchmark score tables linked by the paper.

Only the public data tables are used here. No author implementation is copied.
The tables are LINCS/PRISM-filtered overlap subsets, not the full dataset sizes
listed in the paper. Their released ``score`` column is treated as the target.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ML.feature_engineering.features import (
    DATA_DIR,
    ML_DIR,
    PAIR_SCHEMA_CONCAT,
    PAIR_SCHEMA_SYMMETRIC_SUM_ABSDIFF,
    _load_drug_maps,
    _make_fingerprints,
)
from ML.models.native_fallback import auc_score
from ML.prediction.predict import available_models
from ML.storage import load
from ML.training.train import _metrics

BENCHMARK_DIR = DATA_DIR / "paper_benchmarks"
MODEL_DIR = ML_DIR / "results" / "models"
RESULTS_DIR = ML_DIR / "results"
DATASETS = {
    "ALMANAC": "AlmanacLINCS_PRISM.csv",
    "DrugCombDB": "DrugCombDBLINCS_PRISM.csv",
    "OncologyScreen": "OncologyScreenLINCS_PRISM.csv",
    "ONeil": "OneilLINCS_PRISM.csv",
}


def _context_path() -> Path:
    for suffix in (".joblib", ".pkl"):
        path = MODEL_DIR / f"feature_context{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError("Train models first so the feature context is saved.")


def _load_pair_features(smiles_values: set[str], context: dict):
    """Build structure vectors and map benchmark SMILES to local drug features.

    Fingerprint-vector equality is not a safe compound identifier: equivalent
    SMILES can differ by salt handling, aromatic notation, or fingerprint
    backend. Prefer RDKit canonical SMILES to crosswalk to the local chemical
    table, with exact fingerprint matching retained as a fallback.
    """
    smiles_structures = {value: value for value in smiles_values}
    structure_vectors = _make_fingerprints(smiles_structures)

    fingerprint_candidates: dict[bytes, list[str]] = defaultdict(list)
    for drug, vector in context["fingerprints"].items():
        key = np.asarray(vector, dtype=np.float32).tobytes()
        fingerprint_candidates[key].append(drug)

    signature_ids = {
        "ds": set(context["drug_signatures"]),
        "drs": set(context["resistance_signatures"]),
    }

    # Match canonical molecular structures to the names used by our local
    # chemical table and by the LINCS/GDSC signature dictionaries.
    canonical_candidates: dict[str, list[str]] = defaultdict(list)
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.error")
        try:
            _, local_smiles = _load_drug_maps()
            for drug, structure in local_smiles.items():
                if not structure or structure.strip().lower() in {"#n/a", "n/a", "na", "none", "nan"}:
                    continue
                molecule = Chem.MolFromSmiles(structure)
                if molecule is not None:
                    canonical_candidates[Chem.MolToSmiles(molecule, canonical=True)].append(drug)
            canonical_by_input: dict[str, str] = {}
            for structure in smiles_values:
                if not structure or structure.strip().lower() in {"#n/a", "n/a", "na", "none", "nan"}:
                    continue
                molecule = Chem.MolFromSmiles(structure)
                if molecule is not None:
                    canonical_by_input[structure] = Chem.MolToSmiles(molecule, canonical=True)
        finally:
            RDLogger.EnableLog("rdApp.error")
    except Exception:
        canonical_by_input = {}

    signature_drugs: dict[str, dict[str, str]] = {"ds": {}, "drs": {}}
    for smiles, vector in structure_vectors.items():
        canonical = canonical_by_input.get(smiles)
        matches = canonical_candidates.get(canonical, []) if canonical else []
        if not matches:
            matches = fingerprint_candidates.get(np.asarray(vector, dtype=np.float32).tobytes(), [])
        if not matches:
            continue
        for feature_set, available in signature_ids.items():
            candidates = sorted(set(matches) & available)
            if candidates:
                signature_drugs[feature_set][smiles] = candidates[0]
    return structure_vectors, signature_drugs


def _vectors(frame: pd.DataFrame, feature_set: str, context: dict,
             structure_vectors: dict, signature_drugs: dict,
             pair_schema: str = PAIR_SCHEMA_CONCAT):
    drug_store = {
        "ds": context["drug_signatures"],
        "drs": context["resistance_signatures"],
    }
    left, right, targets = [], [], []
    skipped_drug = skipped_cell = 0
    for drug_a, drug_b, cell_id, target in frame[
        ["drugname1", "drugname2", "cellline", "score"]
    ].itertuples(index=False, name=None):
        cell = context["expression"].get(str(cell_id))
        if cell is None:
            skipped_cell += 1
            continue
        views = ("structure", "ds", "drs") if feature_set == "multilevel" else (feature_set,)
        pairs = []
        for view in views:
            if view == "structure":
                a, b = structure_vectors.get(drug_a), structure_vectors.get(drug_b)
            else:
                map_for_feature = signature_drugs[view]
                drug_store_for_feature = drug_store[view]
                id_a, id_b = map_for_feature.get(drug_a), map_for_feature.get(drug_b)
                a = drug_store_for_feature.get(id_a) if id_a else None
                b = drug_store_for_feature.get(id_b) if id_b else None
            if a is None or b is None:
                pairs = []
                break
            pairs.append((a, b))
        if not pairs:
            skipped_drug += 1
            continue
        # Score both drug orders and average at prediction time because source
        # tables identify compounds by SMILES rather than the training aliases.
        left_parts, right_parts = [], []
        for a, b in pairs:
            if view == "structure" and pair_schema == PAIR_SCHEMA_SYMMETRIC_SUM_ABSDIFF:
                symmetric_parts = (a + b, np.abs(a - b), cell)
                left_parts.extend(symmetric_parts)
                right_parts.extend(symmetric_parts)
            else:
                left_parts.extend((a, b, cell))
                right_parts.extend((b, a, cell))
        left.append(np.concatenate(left_parts).astype(np.float32, copy=False))
        right.append(np.concatenate(right_parts).astype(np.float32, copy=False))
        targets.append(float(target))
    width = int(len(context["expression"][next(iter(context["expression"]))]))
    if feature_set == "multilevel":
        drug_width = int(context["drug_dim"])
        signature_width = int(context["cell_dim"])
        feature_width = (2 * drug_width + width) + 2 * (2 * signature_width + width)
    else:
        drug_width = int(context["drug_dim"]) if feature_set == "structure" else int(context["cell_dim"])
        feature_width = 2 * drug_width + width
    empty = np.empty((0, feature_width), dtype=np.float32)
    return (
        np.vstack(left) if left else empty,
        np.vstack(right) if right else empty.copy(),
        np.asarray(targets, dtype=float),
        skipped_drug,
        skipped_cell,
    )


def _predict_symmetry_averaged(model, left: np.ndarray, right: np.ndarray, batch_size: int = 4096):
    predictions = []
    for start in range(0, len(left), batch_size):
        stop = min(start + batch_size, len(left))
        p_left = np.asarray(model.predict(left[start:stop]), dtype=float)
        p_right = np.asarray(model.predict(right[start:stop]), dtype=float)
        predictions.append((p_left + p_right) / 2.0)
    return np.concatenate(predictions) if predictions else np.asarray([], dtype=float)


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    context = load(_context_path())
    model_bundles = {}
    for item in available_models():
        path = MODEL_DIR / item["file"]
        bundle = load(path)
        model_bundles[(item["feature_set"], item["model"])] = bundle
    if not model_bundles:
        raise FileNotFoundError("No saved models found. Train models first.")

    inputs: dict[str, pd.DataFrame] = {}
    all_smiles: set[str] = set()
    for dataset, filename in DATASETS.items():
        path = BENCHMARK_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing paper-linked benchmark file: {path}")
        frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig", encoding_errors="replace")
        required = {"drugname1", "drugname2", "cellline", "score"}
        if required - set(frame.columns):
            raise ValueError(f"{filename} is missing columns: {sorted(required - set(frame.columns))}")
        frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
        frame = frame.dropna(subset=["drugname1", "drugname2", "cellline", "score"]).reset_index(drop=True)
        inputs[dataset] = frame
        all_smiles.update(frame["drugname1"].astype(str))
        all_smiles.update(frame["drugname2"].astype(str))

    structure_vectors, signature_drugs = _load_pair_features(all_smiles, context)
    metric_rows, coverage_rows = [], []
    requested_feature_sets = sorted({feature for feature, _ in model_bundles})

    for dataset, frame in inputs.items():
        for feature_set in requested_feature_sets:
            candidates = [
                (model_name, bundle)
                for (saved_feature_set, model_name), bundle in model_bundles.items()
                if saved_feature_set == feature_set
            ]
            schemas = sorted({
                bundle.get("pair_schema", PAIR_SCHEMA_CONCAT)
                for _, bundle in candidates
            })
            for pair_schema in schemas:
                left, right, target, skipped_drug, skipped_cell = _vectors(
                    frame,
                    feature_set,
                    context,
                    structure_vectors,
                    signature_drugs,
                    pair_schema=pair_schema,
                )
                base_coverage = {
                    "dataset": dataset, "source_file": DATASETS[dataset],
                    "source_rows": int(len(frame)), "feature_set": feature_set,
                    "pair_schema": pair_schema,
                    "usable_rows": int(len(target)), "rows_skipped_missing_drug_feature": skipped_drug,
                    "rows_skipped_missing_cell_expression": skipped_cell,
                    "score_harmonization": "Released score units are not verified as DrugComb S-score; do not compare metrics across sources.",
                    "feature_matching": "canonical SMILES crosswalk to local chemical records for DS/DRS, exact fingerprint fallback; direct fingerprint from released SMILES for structure",
                }
                coverage_rows.append(base_coverage)
                if not len(target):
                    continue
                for model_name, bundle in candidates:
                    if bundle.get("pair_schema", PAIR_SCHEMA_CONCAT) != pair_schema:
                        continue
