"""Feature utilities for the SynergyX authors' released feature arrays.

This implementation uses the authors' released DRS, substructure and cell
omics arrays as fixed inputs to a project-trained fallback model. It does not
recreate the full SynergyX attention architecture.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from ML.feature_engineering.features import ML_DIR, _load_drug_maps, normalize, resolve_drug


SOURCE_DIR = ML_DIR / "data" / "processed" / "paper_reference"
DRS_FILE = SOURCE_DIR / "DEGsenWithMASK.npy"
SUBSTRUCTURE_FILE = SOURCE_DIR / "drugSmile_drugSubEmbed_2644.npy"
CELL_OMICS_FILE = SOURCE_DIR / "985_cellGraphs_exp_mut_cn_eff_dep_met_4079_genes_norm.npy"


def canonicalize_smiles(value: object) -> str | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True)


def load_author_feature_maps() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load and project the released features in a deterministic way."""
    drs_objects = np.load(DRS_FILE, allow_pickle=True)
    drs_raw = drs_objects.flat[0] if drs_objects.shape else drs_objects.item()
    token_raw = np.load(SUBSTRUCTURE_FILE, allow_pickle=True).item()
    cell_raw = np.load(CELL_OMICS_FILE, allow_pickle=True).item()

    drs_by_smiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for smiles, (values, mask) in drs_raw.items():
        canonical = canonicalize_smiles(smiles)
        if canonical:
            drs_by_smiles[canonical] = (
                np.nan_to_num(np.asarray(values, dtype=np.float32)),
                np.nan_to_num(np.asarray(mask, dtype=np.float32)),
            )

    token_by_smiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    max_token = 0
    for smiles, (token_ids, token_mask) in token_raw.items():
        canonical = canonicalize_smiles(smiles)
        if not canonical:
            continue
        ids = np.asarray(token_ids, dtype=np.int64)
        mask = np.asarray(token_mask, dtype=np.float32)
        token_by_smiles[canonical] = (ids, mask)
        if ids.size:
            max_token = max(max_token, int(ids.max()))

    shared_drugs = sorted(set(drs_by_smiles) & set(token_by_smiles))
    if len(shared_drugs) < 10:
        raise RuntimeError("Too few author-released drugs have both DRS and structure features.")

    token_counts = np.zeros((len(shared_drugs), max_token + 1), dtype=np.float32)
    for i, smiles in enumerate(shared_drugs):
        token_ids, token_mask = token_by_smiles[smiles]
        valid = (token_mask > 0) & (token_ids >= 0)
        if valid.any():
            token_counts[i] = np.bincount(
                token_ids[valid], minlength=max_token + 1
            ).astype(np.float32)

    token_components = min(32, len(shared_drugs) - 1, token_counts.shape[1] - 1)
    token_projection = PCA(n_components=token_components, svd_solver="full")
    token_values = token_projection.fit_transform(token_counts).astype(np.float32)
    token_projection_by_smiles = {
        smiles: token_values[i] for i, smiles in enumerate(shared_drugs)
    }

    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    drug_features: dict[str, np.ndarray] = {}
    for smiles in shared_drugs:
        molecule = Chem.MolFromSmiles(smiles)
        fingerprint = generator.GetFingerprint(molecule)
        fingerprint_bits = np.fromiter(
            (int(value) for value in fingerprint.ToBitString()),
            dtype=np.float32,
            count=1024,
        )
        drs_values, drs_mask = drs_by_smiles[smiles]
        drug_features[smiles] = np.concatenate(
            [drs_values, drs_mask, token_projection_by_smiles[smiles], fingerprint_bits]
        ).astype(np.float32)

    cell_ids = sorted(str(key) for key in cell_raw)
    cell_matrix = np.stack(
        [np.asarray(cell_raw[cell_id], dtype=np.float32).reshape(-1) for cell_id in cell_ids]
    )
    cell_components = min(64, len(cell_ids) - 1, cell_matrix.shape[1] - 1)
    cell_projection = PCA(
        n_components=cell_components,
        svd_solver="randomized",
        random_state=42,
    )
    cell_values = cell_projection.fit_transform(cell_matrix).astype(np.float32)
    cell_features = {cell_id: cell_values[i] for i, cell_id in enumerate(cell_ids)}
    return drug_features, cell_features


def paper_release_pair_vector(
    drug_a: str,
    drug_b: str,
    cell_line: str,
    context: dict,
    bundle: dict,
) -> np.ndarray:
    """Create a serving vector using stored author features and project aliases."""
    smiles_by_id = context.get("smiles")
    if not isinstance(smiles_by_id, dict) or not smiles_by_id:
        _, smiles_by_id = _load_drug_maps()
        context["smiles"] = smiles_by_id

    def feature_for_drug(name: str) -> tuple[str, np.ndarray]:
        canonical_id = resolve_drug(name, context)
        raw_smiles = smiles_by_id.get(canonical_id)
        canonical_smiles = canonicalize_smiles(raw_smiles or name)
        features = bundle.get("drug_features", {}).get(canonical_smiles)
        if features is None:
            raise ValueError(
                f"No author-release DRS and structure features are available for {name}."
            )
        return str(canonical_smiles), np.asarray(features, dtype=np.float32)

    drug_features = [feature_for_drug(drug_a), feature_for_drug(drug_b)]
    drug_features.sort(key=lambda item: item[0])

    cell_features = bundle.get("cell_features", {})
    cell_key = str(cell_line).strip()
    if cell_key not in cell_features:
        cell_key = context.get("cell_aliases", {}).get(normalize(cell_key), cell_key)
    if cell_key not in cell_features:
        raise ValueError(
            f"No author-release multi-omics profile is available for cell line {cell_line}."
        )

    return np.concatenate(
        [drug_features[0][1], drug_features[1][1], np.asarray(cell_features[cell_key], dtype=np.float32)]
    ).astype(np.float32, copy=False)
