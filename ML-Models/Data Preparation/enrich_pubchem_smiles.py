"""Fill missing DrugComb structures from the PubChem CIDs in its chemical table.

This writes a separate, provenance-tracked enrichment file and leaves the
downloaded DrugComb source table unchanged.
"""
from __future__ import annotations

import csv
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rdkit import Chem

ML_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ML_DIR / "data"
CHEMICAL_FILE = DATA_DIR / "drug_chemical_info.csv.xls"
OUTPUT_FILE = DATA_DIR / "pubchem_smiles_enrichment.csv"
INVALID_STRUCTURES = {"", "#n/a", "n/a", "none", "nan", "null"}
PUBCHEM_ROOT = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"


def _valid_smiles(value: object) -> str | None:
    text = str(value or "").strip()
    if text.lower() in INVALID_STRUCTURES:
        return None
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _request(ids: list[str]) -> dict[str, dict[str, str]]:
    """Fetch a batch, splitting failed batches to isolate bad PubChem IDs."""
    time.sleep(0.25)  # Stay under PubChem's public request rate.
    url = f"{PUBCHEM_ROOT}/{','.join(ids)}/property/SMILES,ConnectivitySMILES/JSON"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "DRS-DS-PubChem-structure-enrichment/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.load(response)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        if len(ids) == 1:
            return {}
        midpoint = len(ids) // 2
        left = _request(ids[:midpoint])
        right = _request(ids[midpoint:])
        return left | right

    output = {}
    properties = payload.get("PropertyTable", {}).get("Properties", [])
    for row in properties:
        cid = str(row.get("CID", ""))
        isomeric = _valid_smiles(row.get("SMILES", ""))
        connectivity = _valid_smiles(row.get("ConnectivitySMILES", ""))
        if cid.isdigit() and isomeric:
            output[cid] = {
                "isomeric_smiles": isomeric,
                "connectivity_smiles": connectivity or isomeric,
            }

    missing = [cid for cid in ids if cid not in output]
    if len(ids) > 1 and missing:
        for cid in missing:
            output.update(_request([cid]))
    return output


def enrich() -> dict:
    table = pd.read_csv(
        CHEMICAL_FILE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        encoding_errors="replace",
    )
    placeholder = table["smilesString"].map(
        lambda value: _valid_smiles(value) is None
    )
    missing_rows = table.loc[placeholder]
    cid_by_row: dict[int, list[str]] = {}
    all_cids: set[str] = set()
    for index, row in missing_rows.iterrows():
        cids = list(dict.fromkeys(
            str(int(value)) for value in re.findall(r"\d+", str(row.get("cIds", "")))
        ))
        if cids:
            cid_by_row[int(index)] = cids
            all_cids.update(cids)

    structures: dict[str, dict[str, str]] = {}
    ordered_cids = sorted(all_cids, key=int)
    for start in range(0, len(ordered_cids), 40):
        structures.update(_request(ordered_cids[start:start + 40]))

    retrieved_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for index, cids in cid_by_row.items():
        original = table.loc[index]
        selected_cid = next((cid for cid in cids if cid in structures), None)
        if selected_cid is None:
            continue
        rows.append({
            "drugName": str(original.get("drugName", "")).strip(),
            "drugNameOfficial": str(original.get("drugNameOfficial", "")).strip(),
            "cid": selected_cid,
            "isomeric_smiles": structures[selected_cid]["isomeric_smiles"],
            "connectivity_smiles": structures[selected_cid]["connectivity_smiles"],
            "source": "PubChem PUG REST; CID supplied by DrugComb chemical table",
            "source_url": f"{PUBCHEM_ROOT}/{selected_cid}/property/SMILES,ConnectivitySMILES/JSON",
            "retrieved_at_utc": retrieved_at,
        })

    output = pd.DataFrame(rows).drop_duplicates(subset=["drugName", "cid"])
    output = output.sort_values(["drugName", "cid"], kind="stable")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False, quoting=csv.QUOTE_MINIMAL)
    report = {
        "chemical_table_rows": int(len(table)),
        "placeholder_or_invalid_smiles_rows": int(placeholder.sum()),
        "unique_cids_queried": int(len(ordered_cids)),
        "unique_cids_resolved": int(len(structures)),
        "drug_rows_enriched": int(len(output)),
        "output_file": str(OUTPUT_FILE.relative_to(ML_DIR.parent)),
        "retrieved_at_utc": retrieved_at,
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    enrich()
