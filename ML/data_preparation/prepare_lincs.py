"""Prepare the curated quantitative LINCS Level 5 drug signatures.

The official signatureSearchData resource EH3226 is a 24-hour, 10-uM
curated subset. Its HDF5 matrix stores signatures by rows and Entrez genes
by columns. This script maps those Entrez IDs to symbols and averages
signature profiles by compound to create a compact wide CSV for the ML code.

Run from the project root:
    python -m ML.data_preparation.prepare_lincs
"""
from __future__ import annotations

import csv
import gzip
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

from ML.feature_engineering.features import (
    DATA_DIR, EXPRESSION_FILE, LINCS_GMT_FILE, _load_drug_maps, _read_gmt,
    normalize,
)

ML_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = DATA_DIR / "processed"
H5_FILE = PROCESSED_DIR / "lincs_level5_filtered_EH3226.h5"
MAPPING_FILE = PROCESSED_DIR / "lincs_entrez_gene_symbols.csv"
OUTPUT_FILE = PROCESSED_DIR / "lincs_level5_drug_signatures.csv"
GENE_INFO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/"
    "Homo_sapiens.gene_info.gz"
)
BLOCK_ROWS = 512


def _selected_vocabulary() -> list[str]:
    import pandas as pd

    if not EXPRESSION_FILE.exists() or not LINCS_GMT_FILE.exists():
        raise FileNotFoundError(
            "The DepMap expression CSV and LINCS GMT must be in ML/data before "
            "preparing numeric signatures."
        )
    columns = pd.read_csv(EXPRESSION_FILE, nrows=0, encoding="utf-8-sig").columns
    excluded = {
        "DEPMAP_ID", "CELL_LINE_DISPLAY_NAME", "LINEAGE_1", "LINEAGE_2",
        "LINEAGE_3", "LINEAGE_4", "LINEAGE_5", "LINEAGE_6",
    }
    gene_candidates = {str(x).upper() for x in columns if str(x).upper() not in excluded}
    aliases, _ = _load_drug_maps()
    vocab, _ = _read_gmt(gene_candidates, aliases)
    return vocab


def _fetch_gene_symbols(selected_symbols: set[str], entrez_ids: set[str]) -> dict[str, str]:
    """Fetch official NCBI gene_info once and retain only genes this project uses."""
    if MAPPING_FILE.exists():
        import pandas as pd

        cached = pd.read_csv(MAPPING_FILE, dtype=str, keep_default_na=False)
        mapping = dict(zip(cached["entrez_id"], cached["gene_symbol"]))
        if mapping:
            # This compact cache was written for the fixed project vocabulary.
            return mapping

    request = urllib.request.Request(
        GENE_INFO_URL,
        headers={"User-Agent": "DRS-Drug-Combination-Project/1.0"},
    )
    mapping: dict[str, str] = {}
    with urllib.request.urlopen(request, timeout=60) as response:
        with gzip.GzipFile(fileobj=response) as stream:
            for raw_line in stream:
                fields = raw_line.decode("utf-8", errors="replace").rstrip("\n").split("\t")
                if len(fields) < 3 or fields[0] != "9606":
                    continue
                entrez_id, symbol = fields[1], fields[2].upper()
                if entrez_id in entrez_ids and symbol in selected_symbols:
                    mapping[entrez_id] = symbol

    MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_path = MAPPING_FILE.with_suffix(".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["entrez_id", "gene_symbol"])
        writer.writerows(sorted(mapping.items()))
    temp_path.replace(MAPPING_FILE)
    return mapping


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip("\x00 ")
    return str(value).strip()


def prepare() -> dict[str, int]:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Install h5py with: python -m pip install h5py") from exc
    if not H5_FILE.exists():
        raise FileNotFoundError(
            f"Missing official LINCS HDF5 file: {H5_FILE}\n"
            "Download ExperimentHub EH3226 from the official Bioconductor "
            "signatureSearchData resource, then rerun this command."
        )

    vocab = _selected_vocabulary()
    selected_symbols = set(vocab)
    with h5py.File(H5_FILE, "r") as h5:
        gene_ids = [_decode(item) for item in h5["rownames"][0]]
        profile_names = [_decode(item) for item in h5["colnames"][0]]
        assay = h5["assay"]
        if assay.shape != (len(profile_names), len(gene_ids)):
            raise ValueError(
                f"Unexpected EH3226 matrix orientation: assay={assay.shape}, "
                f"profiles={len(profile_names)}, genes={len(gene_ids)}"
            )

        id_to_symbol = _fetch_gene_symbols(selected_symbols, set(gene_ids))
        positions_by_symbol: dict[str, list[int]] = defaultdict(list)
        for position, entrez_id in enumerate(gene_ids):
            symbol = id_to_symbol.get(entrez_id)
            if symbol in selected_symbols:
                positions_by_symbol[symbol].append(position)
        genes = [gene for gene in vocab if gene in positions_by_symbol]
        if not genes:
            raise ValueError(
                "None of the selected genes mapped to the LINCS Entrez gene IDs. "
                "Check the NCBI annotation download and expression gene symbols."
            )
        positions = sorted(
            position for gene in genes for position in positions_by_symbol[gene]
        )
        position_symbols = [id_to_symbol[gene_ids[position]] for position in positions]
        position_indices = np.asarray(positions, dtype=np.int64)
        symbol_positions: list[np.ndarray] = [
            np.asarray(
                [j for j, symbol in enumerate(position_symbols) if symbol == gene],
                dtype=np.int64,
            )
            for gene in genes
        ]

        aliases, _ = _load_drug_maps()
        profile_drugs = []
        for profile in profile_names:
            perturbagen = profile.split("__", 1)[0]
            profile_drugs.append(aliases.get(normalize(perturbagen), normalize(perturbagen)))

        sums: dict[str, np.ndarray] = {}
        counts: dict[str, np.ndarray] = {}
        signatures_seen = 0
        for start in range(0, len(profile_names), BLOCK_ROWS):
            end = min(start + BLOCK_ROWS, len(profile_names))
            block = np.asarray(assay[start:end, position_indices], dtype=np.float32)
            block[~np.isfinite(block)] = np.nan
            # Multiple Entrez rows can share a current gene symbol. Collapse
            # those rows within each profile before averaging across cell lines.
            collapsed = np.empty((end - start, len(genes)), dtype=np.float32)
            for gene_pos, gene_positions in enumerate(symbol_positions):
                values = block[:, gene_positions]
                valid = np.isfinite(values)
                count = valid.sum(axis=1)
                sums_for_gene = np.nansum(values, axis=1)
                collapsed[:, gene_pos] = np.divide(
                    sums_for_gene, count, out=np.full(len(count), np.nan, dtype=np.float32),
                    where=count > 0,
                )

            local_drugs = profile_drugs[start:end]
            for drug in set(local_drugs):
                if not drug:
                    continue
                local_rows = np.asarray(
                    [i for i, item in enumerate(local_drugs) if item == drug],
                    dtype=np.int64,
                )
                values = collapsed[local_rows]
                valid = np.isfinite(values)
                if drug not in sums:
                    sums[drug] = np.zeros(len(genes), dtype=np.float64)
                    counts[drug] = np.zeros(len(genes), dtype=np.int64)
                sums[drug] += np.nansum(values, axis=0)
                counts[drug] += valid.sum(axis=0)
                signatures_seen += len(local_rows)

    if not sums:
        raise ValueError("No LINCS perturbagens were converted.")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_path = OUTPUT_FILE.with_suffix(".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["drug_name", *genes])
        for drug in sorted(sums):
            vector = np.divide(
                sums[drug], counts[drug],
                out=np.zeros(len(genes), dtype=np.float64),
                where=counts[drug] > 0,
            )
            writer.writerow([drug, *[f"{value:.8g}" for value in vector]])
    temp_path.replace(OUTPUT_FILE)
    stats = {
        "profiles_in_source": len(profile_names),
        "profiles_aggregated": signatures_seen,
        "genes_selected": len(vocab),
        "genes_mapped": len(genes),
        "drugs_written": len(sums),
    }
    print("Prepared quantitative LINCS signatures:")
    for key, value in stats.items():
        print(f"  {key.replace('_', ' ')}: {value:,}")
    print(f"  saved: {OUTPUT_FILE}")
    print("DS aggregation: mean numeric signature across available LINCS cell lines per compound.")
    return stats


if __name__ == "__main__":
    prepare()

