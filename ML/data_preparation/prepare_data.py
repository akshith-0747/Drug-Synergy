import csv
import argparse
import math
import re
from pathlib import Path

import pandas as pd

from ML.data_preparation.identifier_maps import (
    load_official_cell_aliases,
    load_official_drug_maps,
)

ML_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ML_DIR / "data"
OUTPUT_DIR = DATA_DIR / "processed"

SOURCE_FILE = DATA_DIR / "summary_table_v1.4.csv"
DRUG_FILE = DATA_DIR / "drug_chemical_info.csv.xls"
PUBCHEM_ENRICHMENT_FILE = DATA_DIR / "pubchem_smiles_enrichment.csv"
EXPRESSION_FILE = DATA_DIR / "Expression_(Short-read)_Public_26Q1_subsetted.csv"
OUTPUT_FILE = OUTPUT_DIR / "matched_drugcomb.csv"


def normalize(value):
    """Make names and IDs comparable by ignoring spaces and punctuation."""
    return re.sub(r"[^A-Z0-9]", "", str(value).strip().upper())


def add_alias(aliases, conflicts, alias, smiles):
    key = normalize(alias)
    structure = str(smiles or "").strip()
    if (
        not key
        or not structure
        or structure.lower() in {"#n/a", "n/a", "none", "nan", "null"}
        or key in conflicts
    ):
        return

    previous = aliases.get(key)
    if previous is not None and previous != smiles:
        aliases.pop(key, None)
        conflicts.add(key)
    else:
        aliases[key] = structure


def load_drug_smiles():
    """Map drug names and PubChem IDs to SMILES structures."""
    aliases = {}
    conflicts = set()

    with DRUG_FILE.open(
        "r", encoding="utf-8-sig", errors="replace", newline=""
    ) as file:
        for row in csv.DictReader(file):
            smiles = row.get("smilesString", "").strip()
            if smiles.lower() in {"#n/a", "n/a", "none", "nan", "null"}:
                continue
            if not smiles:
                continue

            add_alias(aliases, conflicts, row.get("drugName", ""), smiles)
            add_alias(
                aliases, conflicts, row.get("drugNameOfficial", ""), smiles
            )

            # The dataset may use numeric PubChem IDs as drug names.
            cids = row.get("cIds", "")
            for cid in re.findall(r"\d+", cids):
                add_alias(aliases, conflicts, str(int(cid)), smiles)

    if PUBCHEM_ENRICHMENT_FILE.exists():
        with PUBCHEM_ENRICHMENT_FILE.open(
            "r", encoding="utf-8-sig", errors="replace", newline=""
        ) as file:
            for row in csv.DictReader(file):
                smiles = row.get("isomeric_smiles", "").strip()
                add_alias(aliases, conflicts, row.get("drugName", ""), smiles)
                add_alias(aliases, conflicts, row.get("drugNameOfficial", ""), smiles)
                cid = row.get("cid", "").strip()
                if cid.isdigit():
                    add_alias(aliases, conflicts, str(int(cid)), smiles)

    official_aliases, official_smiles = load_official_drug_maps()
    for alias, canonical in official_aliases.items():
        if alias in aliases or alias in conflicts:
            continue
        smiles = official_smiles.get(canonical, "")
        if smiles:
            aliases[alias] = smiles

    return aliases


def load_expression_cell_ids():
    """Map expression-file cell-line names to DepMap IDs."""
    name_to_id = {}
    ambiguous_names = set()

    with EXPRESSION_FILE.open(
        "r", encoding="utf-8-sig", errors="replace", newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            name = normalize(row.get("cell_line_display_name", ""))
            depmap_id = row.get("depmap_id", "").strip()

            if not name or not depmap_id or name in ambiguous_names:
                continue

            previous = name_to_id.get(name)
            if previous is not None and previous != depmap_id:
                name_to_id.pop(name, None)
                ambiguous_names.add(name)
            else:
                name_to_id[name] = depmap_id

    # DrugComb's own crosswalk supplies aliases and model IDs that are absent
    # from the expression display-name column. Add only IDs present in the
    # expression matrix, and only aliases that resolve to one such ID.
    official_aliases = load_official_cell_aliases(set(name_to_id.values()))
    for alias, depmap_id in official_aliases.items():
        name_to_id.setdefault(alias, depmap_id)

    return name_to_id


def main(
    source_file: Path = SOURCE_FILE,
    target_column: str = "S",
    output_file: Path = OUTPUT_FILE,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    drug_smiles = load_drug_smiles()
    cell_ids = load_expression_cell_ids()

    preserved_score_columns = [
        "S_mean",
        "S_sum",
        "S_max",
        "css",
        "css_ri",
        "synergy_zip",
        "synergy_bliss",
        "synergy_loewe",
        "synergy_hsa",
    ]
    output_columns = [
        "block_id",
        "source_dataset",
        "source_study",
        "drug_row",
        "drug_col",
        "cell_line_name",
        "depmap_id",
        "smiles_row",
        "smiles_col",
        "S",
        *preserved_score_columns,
    ]

    total = written = 0
    missing_score = missing_cell = missing_structure = 0

    with source_file.open(
        "r", encoding="utf-8-sig", errors="replace", newline=""
    ) as input_file:
        reader = csv.DictReader(input_file)

        required_columns = {
            "block_id",
            "drug_row",
            "drug_col",
            "cell_line_name",
            target_column,
        }
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"{source_file.name} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        with output_path.open(
            "w", encoding="utf-8", newline=""
        ) as output_stream:
            writer = csv.DictWriter(output_stream, fieldnames=output_columns)
            writer.writeheader()

            for row in reader:
                total += 1

                try:
                    s_score = float(row.get(target_column, ""))
                    if not math.isfinite(s_score):
                        raise ValueError
                except (TypeError, ValueError):
                    missing_score += 1
                    continue

                depmap_id = cell_ids.get(
                    normalize(row.get("cell_line_name", ""))
                )
                if not depmap_id:
                    missing_cell += 1
                    continue

                smiles_row = drug_smiles.get(
                    normalize(row.get("drug_row", ""))
                )
                smiles_col = drug_smiles.get(
                    normalize(row.get("drug_col", ""))
                )
                if not smiles_row or not smiles_col:
                    missing_structure += 1
                    continue

                output_row = {
                    column: row.get(column, "")
                    for column in preserved_score_columns
                }
                output_row.update({
                    "block_id": row.get("block_id", "") or f"row-{total}",
                    "source_dataset": source_file.stem,
                    "source_study": row.get("study_name", ""),
                    "drug_row": row.get("drug_row", ""),
                    "drug_col": row.get("drug_col", ""),
                    "cell_line_name": row.get("cell_line_name", ""),
                    "depmap_id": depmap_id,
                    "smiles_row": smiles_row,
                    "smiles_col": smiles_col,
                    "S": s_score,
                })
                writer.writerow(output_row)
                written += 1

    print(f"Input rows: {total:,}")
    print(f"Matched rows written: {written:,}")
    print(f"Rows skipped for missing S score: {missing_score:,}")
    print(f"Rows skipped for missing cell-line match: {missing_cell:,}")
    print(f"Rows skipped for missing drug structure: {missing_structure:,}")
    print(f"Target column: {target_column}")
    print(f"Saved file: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare a DrugComb table with mapped cell and drug features."
    )
    parser.add_argument("--source-file", type=Path, default=SOURCE_FILE)
    parser.add_argument("--target-column", default="S")
    parser.add_argument("--output-file", type=Path, default=OUTPUT_FILE)
    arguments = parser.parse_args()
    main(
        source_file=arguments.source_file,
        target_column=arguments.target_column,
        output_file=arguments.output_file,
    )
