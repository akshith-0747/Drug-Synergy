"""Resolve DrugComb names through the official v1.4 identifier crosswalk."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re

import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ML_DIR / "data"
DRUG_IDENTIFIERS_FILE = (
    DATA_DIR / "raw" / "drugcomb_v1_4_identifiers"
    / "DrugComb_drug_identifiers.xlsx"
)
CELL_IDENTIFIERS_FILE = (
    DATA_DIR / "raw" / "drugcomb_v1_4_identifiers"
    / "DrugComb_cell_line_identifiers.xlsx"
)


def normalize(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").strip().upper())


def _valid_smiles(value: object) -> str:
    smiles = str(value or "").strip()
    return "" if smiles.lower() in {"", "#n/a", "n/a", "none", "nan", "null"} else smiles


def load_official_drug_maps(
    existing_aliases: dict[str, str] | None = None,
    existing_smiles: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return unambiguous official alias->canonical and canonical->SMILES maps.

    Existing project canonical names are preferred where a current alias maps
    unambiguously to the same official InChIKey. Otherwise the official DrugComb
    display name is used. Ambiguous aliases are omitted rather than guessed.
    """
    if not DRUG_IDENTIFIERS_FILE.exists():
        return {}, {}

    table = pd.read_excel(
        DRUG_IDENTIFIERS_FILE,
        sheet_name="drug",
        dtype=str,
        keep_default_na=False,
    )
    alias_identities: dict[str, set[str]] = defaultdict(set)
    identity_names: dict[str, str] = {}
    identity_smiles: dict[str, str] = {}

    for row in table.to_dict("records"):
        smiles = _valid_smiles(row.get("smiles", ""))
        if not smiles:
            continue
        inchikey = str(row.get("inchikey", "")).strip()
        identity = inchikey or smiles
        display = normalize(row.get("dname", "")) or identity
        identity_names.setdefault(identity, display)
        identity_smiles.setdefault(identity, smiles)

        names = [
            row.get("dname", ""),
            row.get("cid", ""),
            row.get("cid_m", ""),
            row.get("cid_s", ""),
            row.get("chembl_id", ""),
            row.get("drugbank_id", ""),
            row.get("kegg_id", ""),
            row.get("stitch_name", ""),
        ]
        names.extend(str(row.get("synonyms", "")).split(";"))
        for raw_name in names:
            key = normalize(raw_name)
            if key and key not in {"NA", "NAN", "NONE", "NULL"}:
                alias_identities[key].add(identity)

    preferred_names: dict[str, Counter[str]] = defaultdict(Counter)
    for alias, canonical in (existing_aliases or {}).items():
        identities = alias_identities.get(alias, set())
        if len(identities) == 1:
            identity = next(iter(identities))
            if canonical in (existing_smiles or {}):
                preferred_names[identity][canonical] += 1

    identity_to_canonical: dict[str, str] = {}
    for identity, display in identity_names.items():
        preferred = preferred_names.get(identity)
        identity_to_canonical[identity] = (
            sorted(preferred.items(), key=lambda item: (-item[1], item[0]))[0][0]
            if preferred
            else display
        )

    # Do not collapse two different InChIKeys onto one existing label.
    canonical_identities: dict[str, set[str]] = defaultdict(set)
    for identity, canonical in identity_to_canonical.items():
        canonical_identities[canonical].add(identity)
    for canonical, identities in canonical_identities.items():
        if len(identities) > 1:
            for identity in identities:
                identity_to_canonical[identity] = identity_names[identity]

    alias_to_canonical: dict[str, str] = {}
    for alias, identities in alias_identities.items():
        if len(identities) == 1:
            identity = next(iter(identities))
            alias_to_canonical[alias] = identity_to_canonical[identity]

    canonical_to_smiles: dict[str, str] = {}
    for identity, canonical in identity_to_canonical.items():
        current = _valid_smiles((existing_smiles or {}).get(canonical, ""))
        canonical_to_smiles.setdefault(
            canonical,
            current or identity_smiles[identity],
        )

    return alias_to_canonical, canonical_to_smiles


def load_official_cell_aliases(
    expression_depmap_ids: set[str],
) -> dict[str, str]:
    """Map official DrugComb cell-name aliases only to available expression IDs."""
    if not CELL_IDENTIFIERS_FILE.exists():
        return {}

    table = pd.read_excel(
        CELL_IDENTIFIERS_FILE,
        sheet_name="cell_line",
        dtype=str,
        keep_default_na=False,
    )
    candidates: dict[str, set[str]] = defaultdict(set)
    for row in table.to_dict("records"):
        depmap_id = str(row.get("depmap_id", "")).strip()
        if not depmap_id or depmap_id not in expression_depmap_ids:
            continue
        names = [
            row.get("name", ""),
            row.get("ccle_name", ""),
            row.get("cosmic_id", ""),
            row.get("cellosaurus_accession", ""),
        ]
        names.extend(str(row.get("synonyms", "")).split(";"))
        for raw_name in names:
            key = normalize(raw_name)
            if key and key not in {"NA", "NAN", "NONE", "NULL"}:
                candidates[key].add(depmap_id)

    return {
        alias: next(iter(depmap_ids))
        for alias, depmap_ids in candidates.items()
        if len(depmap_ids) == 1
    }
