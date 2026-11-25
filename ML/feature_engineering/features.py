"""Feature construction for DrugComb S-score prediction.

DS uses quantitative LINCS Level 5 signatures when prepared, with a signed GMT
proxy available only when the numeric LINCS table is absent. DRS is computed
from GDSC monotherapy IC50 and baseline cell-line RNA expression.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ML.data_preparation.identifier_maps import (
    load_official_cell_aliases,
    load_official_drug_maps,
)

ML_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ML_DIR / "data"

CHEMICAL_FILE = DATA_DIR / "drug_chemical_info.csv.xls"
PUBCHEM_ENRICHMENT_FILE = DATA_DIR / "pubchem_smiles_enrichment.csv"
EXPRESSION_FILE = DATA_DIR / "Expression_(Short-read)_Public_26Q1_subsetted.csv"
LINCS_GMT_FILE = DATA_DIR / "LINCS_XMT_2022-12-13_LINCS_L1000_Chem_Pert_Consensus_Sigs.gmt"
GDSC_FILE = DATA_DIR / "summary_gdsc_integrated.csv.csv"
GDSC1_RESPONSE_FILE = DATA_DIR / "GDSC1_fitted_dose_response_25Feb20.csv"
GDSC2_RESPONSE_FILE = DATA_DIR / "GDSC2_fitted_dose_response_25Feb20.csv"
PRISM_DOSE_RESPONSE_FILE = DATA_DIR / "raw" / "prism_rep" / "secondary-screen-dose-response-curve-parameters.csv"

LINCS_NUMERIC_FILE = DATA_DIR / "processed" / "lincs_level5_signatures.csv"
LINCS_WIDE_FILE = DATA_DIR / "processed" / "lincs_level5_drug_signatures.csv"

PAIR_SCHEMA_CONCAT = "concat_v1"
PAIR_SCHEMA_SYMMETRIC_SUM_ABSDIFF = "symmetric_sum_absdiff_v1"


# ============================================================
# BASIC NORMALIZATION
# ============================================================

def normalize(value: object) -> str:
    """Normalize drug/cell-line names for matching."""
    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(value or "").strip().upper()
    )


# ============================================================
# DRUG MAPS
# ============================================================

def _load_drug_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Return alias-to-canonical IDs and canonical-ID-to-SMILES maps."""

    aliases: dict[str, str] = {}
    conflicts: set[str] = set()
    smiles: dict[str, str] = {}

    table = pd.read_csv(
        CHEMICAL_FILE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        encoding_errors="replace",
    )

    for row in table.to_dict("records"):

        canonical = normalize(
            row.get("drugName")
            or row.get("drugNameOfficial")
        )

        structure = str(row.get("smilesString", "")).strip()
        if structure.lower() in {"", "#n/a", "n/a", "none", "nan", "null"}:
            structure = ""

        if not canonical:
            continue

        if structure:
            old = smiles.get(canonical)

            if old is None or old == structure:
                smiles[canonical] = structure

        possible = [
            row.get("drugName", ""),
            row.get("drugNameOfficial", ""),
        ]

        possible.extend(
            re.findall(
                r"\d+",
                row.get("cIds", "")
            )
        )

        for raw_alias in possible:

            alias = normalize(raw_alias)

            if not alias or alias in conflicts:
                continue

            old = aliases.get(alias)

            if old is not None and old != canonical:
                aliases.pop(alias, None)
                conflicts.add(alias)

            else:
                aliases[alias] = canonical

        aliases.setdefault(
            canonical,
            canonical
        )

    if PUBCHEM_ENRICHMENT_FILE.exists():
        enrichment = pd.read_csv(
            PUBCHEM_ENRICHMENT_FILE,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        for row in enrichment.to_dict("records"):
            canonical = normalize(row.get("drugName", ""))
            structure = str(row.get("isomeric_smiles", "")).strip()
            if not canonical or structure.lower() in {
                "", "#n/a", "n/a", "none", "nan", "null"
            }:
                continue
            # Keep an existing valid source structure if one was supplied.
            smiles.setdefault(canonical, structure)
            aliases.setdefault(canonical, canonical)

    official_aliases, official_smiles = load_official_drug_maps(
        aliases,
        smiles,
    )
    for canonical, structure in official_smiles.items():
        smiles.setdefault(canonical, structure)
        aliases.setdefault(canonical, canonical)
    for alias, canonical in official_aliases.items():
        if alias not in conflicts:
            aliases.setdefault(alias, canonical)

    return aliases, smiles


def _canonical_drug(
    value: object,
    aliases: dict[str, str]
) -> str:
    key = normalize(value)
    return aliases.get(key, key)


# ============================================================
# LINCS GMT
# ============================================================

def _read_gmt(
    gene_candidates: set[str],
    aliases: dict[str, str],
    max_genes: int = 256
):
    counts: Counter[str] = Counter()

    parsed: list[
        tuple[str, str, list[str]]
    ] = []

    with LINCS_GMT_FILE.open(
        "r",
        encoding="utf-8",
        errors="replace"
    ) as stream:

        for line in stream:

            fields = line.rstrip(
                "\r\n"
            ).split("\t")

            if len(fields) < 3:
                continue

            match = re.match(
                r"^(.*?)\s+(UP|DOWN)$",
                fields[0],
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            drug_key = _canonical_drug(
                match.group(1),
                aliases
            )

            direction = match.group(2).lower()

            genes = [
                gene.strip().upper()
                for gene in fields[2:]
                if gene.strip()
            ]

            parsed.append(
                (
                    drug_key,
                    direction,
                    genes
                )
            )

            counts.update(
                gene
                for gene in genes
                if gene in gene_candidates
            )

    vocab = [
        gene
        for gene, _ in counts.most_common(max_genes)
    ]

    if not vocab:
        raise ValueError(
            "No LINCS GMT genes overlap the expression file's gene columns."
        )

    index = {
        gene: i
        for i, gene in enumerate(vocab)
    }

    signed: dict[str, np.ndarray] = {}

    for drug, direction, genes in parsed:

        vector = signed.setdefault(
            drug,
            np.zeros(
                len(vocab),
                dtype=np.float32
            )
        )

        sign = (
            1.0
            if direction == "up"
            else -1.0
        )

        for gene in genes:

            idx = index.get(gene)

            if idx is not None:
                vector[idx] += sign

    for vector in signed.values():
        np.clip(
            vector,
            -1.0,
            1.0,
            out=vector
        )

    return vocab, signed


# ============================================================
# NUMERIC LINCS
# ============================================================

def _read_numeric_lincs(
    vocab: list[str],
    aliases: dict[str, str]
) -> dict[str, np.ndarray]:

    """Read drug-averaged quantitative LINCS signatures or legacy long CSV."""

    # --------------------------------------------------------
    # Wide LINCS file
    # --------------------------------------------------------

    if LINCS_WIDE_FILE.exists():

        header = pd.read_csv(
            LINCS_WIDE_FILE,
            nrows=0,
            encoding="utf-8-sig"
        ).columns.tolist()

        if "drug_name" not in header:
            raise ValueError(
                f"{LINCS_WIDE_FILE.name} must contain a drug_name column."
            )

        genes = [
            gene
            for gene in vocab
            if gene in header
        ]

        if not genes:
            raise ValueError(
                f"No selected signature genes occur in {LINCS_WIDE_FILE.name}."
            )

        frame = pd.read_csv(
            LINCS_WIDE_FILE,
            usecols=["drug_name", *genes],
            encoding="utf-8-sig",
            low_memory=False,
        )

        drug_keys = frame["drug_name"].map(
            lambda value: _canonical_drug(
                value,
                aliases
            )
        )

        values = frame[genes].apply(
            pd.to_numeric,
            errors="coerce"
        )

        means = values.groupby(
            drug_keys,
            sort=False
        ).mean()

        output: dict[str, np.ndarray] = {}

        gene_index = {
            gene: index
            for index, gene in enumerate(vocab)
        }

        for drug, row in means.iterrows():

            vector = np.zeros(
                len(vocab),
                dtype=np.float32
            )

            for gene, value in row.items():

                if pd.notna(value):
                    vector[
                        gene_index[gene]
                    ] = float(value)

            output[drug] = vector

        return output

    # --------------------------------------------------------
    # Long LINCS file
    # --------------------------------------------------------

    if not LINCS_NUMERIC_FILE.exists():
        return {}

    header = pd.read_csv(
        LINCS_NUMERIC_FILE,
        nrows=0,
        encoding="utf-8-sig"
    ).columns.tolist()

    required = {
        "drug_name",
        "gene_symbol",
        "score",
    }

    missing = required - set(header)

    if missing:
        raise ValueError(
            f"{LINCS_NUMERIC_FILE.name} is missing columns: {sorted(missing)}"
        )

    metadata = [
        name
        for name in (
            "pert_time",
            "pert_dose"
        )
        if name in header
    ]

    frame = pd.read_csv(
        LINCS_NUMERIC_FILE,
        usecols=[
            "drug_name",
            "gene_symbol",
            "score",
            *metadata,
        ],
        encoding="utf-8-sig",
        encoding_errors="replace",
    )

    if "pert_time" in frame:

        time_text = (
            frame["pert_time"]
            .astype(str)
            .str.lower()
            .str.replace(
                " ",
                "",
                regex=False
            )
        )

        frame = frame[
            time_text.str.contains(
                r"(^|[^0-9])24([^0-9]|$)",
                regex=True
            )
            &
            time_text.str.contains(
                r"h|hr",
                regex=True
            )
        ]

    if "pert_dose" in frame:

        dose_text = (
            frame["pert_dose"]
            .astype(str)
            .str.lower()
            .str.replace(
                "Î¼",
                "u",
                regex=False
            )
            .str.replace(
                "Âµ",
                "u",
                regex=False
            )
            .str.replace(
                " ",
                "",
                regex=False
            )
        )

        frame = frame[
            dose_text.str.match(
                r"^10(?:\.0+)?(?:u?m)?$",
                na=False
            )
        ]

    frame["gene_symbol"] = (
        frame["gene_symbol"]
        .astype(str)
        .str.upper()
    )

    frame["score"] = pd.to_numeric(
        frame["score"],
        errors="coerce"
    )

    frame = frame.dropna(
        subset=["score"]
    )

    frame = frame[
        frame["gene_symbol"].isin(vocab)
    ]

    if frame.empty:
        return {}

    frame["drug_key"] = frame["drug_name"].map(
        lambda value: _canonical_drug(
            value,
            aliases
        )
    )

    means = (
        frame
        .groupby(
            [
                "drug_key",
                "gene_symbol"
            ],
            sort=False
        )["score"]
        .mean()
    )

    gene_index = {
        gene: index
        for index, gene in enumerate(vocab)
    }

    output: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(
            len(vocab),
            dtype=np.float32
        )
    )

    for (
        drug,
        gene
    ), value in means.items():

        output[drug][
            gene_index[gene]
        ] = float(value)

    return dict(output)


# ============================================================
# EXPRESSION
# ============================================================

def _load_expression(vocab: list[str]):

    """Read just the selected genes from the wide DepMap expression matrix."""

    columns = pd.read_csv(
        EXPRESSION_FILE,
        nrows=0,
        encoding="utf-8-sig"
    ).columns.tolist()

    present = [
        gene
        for gene in vocab
        if gene in columns
    ]

    if not present:
        raise ValueError(
            "None of the selected signature genes occur in the expression matrix."
        )

    metadata = [
        name
        for name in (
            "depmap_id",
            "cell_line_display_name"
        )
        if name in columns
    ]

    frame = pd.read_csv(
        EXPRESSION_FILE,
        usecols=metadata + present,
        dtype={
            "depmap_id": str,
            "cell_line_display_name": str,
        },
        encoding="utf-8-sig",
        low_memory=False,
    )

    frame = frame.dropna(
        subset=["depmap_id"]
    )

    for gene in present:

        frame[gene] = pd.to_numeric(
            frame[gene],
            errors="coerce"
        )

        frame[gene] = frame[gene].fillna(
            frame[gene].median()
        )

    values = frame[present].to_numpy(
        dtype=np.float32
    )

    means = np.nanmean(
        values,
        axis=0
    )

    stds = np.nanstd(
        values,
        axis=0
    )

    stds[stds < 1e-6] = 1.0

    values = np.nan_to_num(
        (values - means) / stds,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    expression: dict[str, np.ndarray] = {}
    cell_aliases: dict[str, str] = {}

    for index, row in frame.iterrows():

        depmap_id = str(
            row["depmap_id"]
        ).strip()

        expression[depmap_id] = values[
            frame.index.get_loc(index)
        ]

        display = normalize(
            row.get(
                "cell_line_display_name",
                ""
            )
        )

        if display:
            cell_aliases.setdefault(
                display,
                depmap_id
            )

        cell_aliases.setdefault(
            normalize(depmap_id),
            depmap_id
        )

    official_cell_aliases = load_official_cell_aliases(set(expression))
    for alias, depmap_id in official_cell_aliases.items():
        cell_aliases.setdefault(alias, depmap_id)

    return (
        expression,
        cell_aliases,
        present
    )


# ============================================================
# DRS
# ============================================================

def _load_gdsc_resistance_signatures(
    vocab: list[str],
    aliases: dict[str, str],
    expression: dict[str, np.ndarray],
    cell_aliases: dict[str, str],
) -> dict[str, np.ndarray]:

    """Build per-drug resistant-minus-sensitive expression signatures."""

    frame = pd.read_csv(
        GDSC_FILE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        encoding_errors="replace",
    )

    records: dict[
        str,
        dict[
            str,
            dict[
                int,
                list[float]
            ]
        ]
    ] = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(list)
        )
    )

    # Prefer the full fitted monotherapy tables. The older project-local table
    # is retained as a lower-priority fallback for drug/cell pairs absent from
    # the full GDSC release.
    response_sources = [
        (0, GDSC2_RESPONSE_FILE),
        (1, GDSC1_RESPONSE_FILE),
    ]
    loaded_response_sources = []
    for priority, path in response_sources:
        if not path.exists():
            continue
        table = pd.read_csv(
            path,
            usecols=["CELL_LINE_NAME", "DRUG_NAME", "LN_IC50"],
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
            encoding_errors="replace",
            low_memory=False,
        )
        for cell_name, drug_name, raw_value in table.itertuples(
            index=False,
            name=None,
        ):
            cell_id = cell_aliases.get(normalize(cell_name))
            if not cell_id or cell_id not in expression:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value):
                continue
            drug = _canonical_drug(drug_name, aliases)
            records[drug][cell_id][priority].append(value)
        loaded_response_sources.append(path.name)

    metrics = [
        ("GDSC2_ROW_LN_IC50", "drug_row"),
        ("GDSC2_COL_LN_IC50", "drug_col"),
        ("GDSC1_ROW_LN_IC50", "drug_row"),
        ("GDSC1_COL_LN_IC50", "drug_col"),
    ]

    for row in frame.to_dict("records"):

        cell_id = cell_aliases.get(
            normalize(
                row.get(
                    "cell_line_name",
                    ""
                )
            )
        )

        if not cell_id or cell_id not in expression:
            continue

        legacy_priority_offset = 2 if loaded_response_sources else 0
        for legacy_priority, (
            column,
            drug_column
        ) in enumerate(metrics):

            raw = row.get(
                column,
                ""
            )

            if not raw:
                continue

            try:
                value = float(raw)
            except (
                TypeError,
                ValueError
            ):
                continue

            if np.isfinite(value):

                drug = _canonical_drug(
                    row.get(
                        drug_column,
                        ""
                    ),
                    aliases
                )

                records[
                    drug
                ][
                    cell_id
                ][
                    legacy_priority_offset + legacy_priority
                ].append(value)

    result: dict[str, np.ndarray] = {}

    for drug, cell_values in records.items():

        per_cell: dict[str, float] = {}

        for cell_id, priority_values in cell_values.items():

            # Prefer GDSC2.
            # Fall back to GDSC1.

            if 0 in priority_values:
                priority = 0
            elif 1 in priority_values:
                priority = 1
            elif 2 in priority_values:
                priority = 2
            else:
                priority = 3

            values = priority_values.get(
                priority,
                []
            )

            if values:
                per_cell[cell_id] = float(
                    np.median(values)
                )

        if len(per_cell) < 10:
            continue

        median_ic50 = float(
            np.median(
                list(per_cell.values())
            )
        )

        sensitive = [
            expression[c]
            for c, value in per_cell.items()
            if value < median_ic50
        ]

        resistant = [
            expression[c]
            for c, value in per_cell.items()
            if value >= median_ic50
        ]

        if not sensitive or not resistant:
            continue

        result[drug] = np.asarray(
            np.mean(
                resistant,
                axis=0
            )
            -
            np.mean(
                sensitive,
                axis=0
            ),
            dtype=np.float32,
        )

    return result


def _load_prism_resistance_signatures(
    aliases: dict[str, str],
    expression: dict[str, np.ndarray],
    min_r2: float = 0.5,
    min_cell_lines: int = 10,
) -> tuple[dict[str, np.ndarray], dict]:
    """Build DRS-like expression signatures from quality-filtered PRISM curves.

    This source is intended as a fallback for drugs missing a GDSC signature.
    PRISM and GDSC response values are never pooled because they come from
    different monotherapy assays and use different response measurements.
    """
    if not PRISM_DOSE_RESPONSE_FILE.exists():
        return {}, {
            "available": False,
            "source_file": str(PRISM_DOSE_RESPONSE_FILE),
            "reason": "PRISM dose-response source file is not present",
            "signature_drugs": 0,
        }

    frame = pd.read_csv(
        PRISM_DOSE_RESPONSE_FILE,
        usecols=[
            "name", "depmap_id", "screen_id", "r2", "ic50",
            "passed_str_profiling",
        ],
        dtype=str,
        encoding="utf-8-sig",
        encoding_errors="replace",
        low_memory=False,
    )
    frame["r2_numeric"] = pd.to_numeric(frame["r2"], errors="coerce")
    frame["ic50_numeric"] = pd.to_numeric(frame["ic50"], errors="coerce")
    frame = frame.loc[
        frame["passed_str_profiling"].eq("TRUE")
        & np.isfinite(frame["r2_numeric"])
        & frame["r2_numeric"].ge(min_r2)
        & np.isfinite(frame["ic50_numeric"])
        & frame["ic50_numeric"].gt(0)
        & frame["depmap_id"].isin(expression)
    ].copy()
    frame["drug"] = frame["name"].map(
        lambda value: _canonical_drug(value, aliases)
    )
    frame["screen_rank"] = np.where(frame["screen_id"].eq("MTS010"), 0, 1)
    frame = frame.sort_values(
        ["drug", "depmap_id", "screen_rank", "r2_numeric"],
        ascending=[True, True, True, False],
        kind="mergesort",
    ).drop_duplicates(["drug", "depmap_id"], keep="first")
    frame["log_ic50"] = np.log10(frame["ic50_numeric"].to_numpy(dtype=float))

    signatures: dict[str, np.ndarray] = {}
    cell_counts: dict[str, int] = {}
    for drug, drug_rows in frame.groupby("drug", sort=True):
        if not drug:
            continue
        per_cell = drug_rows.set_index("depmap_id")["log_ic50"]
        if len(per_cell) < min_cell_lines:
            continue
        response_median = float(per_cell.median())
        sensitive_ids = per_cell.index[per_cell < response_median]
        resistant_ids = per_cell.index[per_cell >= response_median]
        if not len(sensitive_ids) or not len(resistant_ids):
            continue
        sensitive = [expression[cell_id] for cell_id in sensitive_ids]
        resistant = [expression[cell_id] for cell_id in resistant_ids]
        signatures[drug] = np.asarray(
            np.mean(resistant, axis=0) - np.mean(sensitive, axis=0),
            dtype=np.float32,
        )
        cell_counts[drug] = int(len(per_cell))

    return signatures, {
        "available": True,
        "source_file": str(PRISM_DOSE_RESPONSE_FILE),
        "source_url": "https://depmap.org/repurposing/",
        "response_measure": "log10(IC50 in micromolar), PRISM secondary screen",
        "quality_filter": {
            "passed_str_profiling": True,
            "dose_response_r2_minimum": float(min_r2),
            "minimum_matched_expression_cell_lines_per_drug": int(min_cell_lines),
            "screen_priority": "MTS010 when a quality-passing curve exists; otherwise highest R2",
        },
        "quality_passing_drug_cell_curves": int(len(frame)),
        "signature_drugs": int(len(signatures)),
        "median_cell_lines_per_signature": (
            float(np.median(list(cell_counts.values()))) if cell_counts else 0.0
        ),
    }


# ============================================================
# FINGERPRINTS
# ============================================================

def _make_fingerprints(
    smiles: dict[str, str],
    n_bits: int = 1024,
) -> dict[str, np.ndarray]:
    """Create fingerprints.

    IMPORTANT:
    This function ALWAYS returns a dictionary.

    That is required because augment_structure_features()
    uses:

        context["fingerprints"].update(
            _make_fingerprints(...)
        )

    RDKit Morgan fingerprints are preferred.

    If RDKit cannot load because Windows/Security policy blocks
    its native DLLs, the project's pure-Python fallback is used.
    """

    # --------------------------------------------------------
    # Try RDKit
    # --------------------------------------------------------

    try:

        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator

        generator = (
            rdFingerprintGenerator
            .GetMorganGenerator(
                radius=2,
                fpSize=n_bits
            )
        )

        output: dict[
            str,
            np.ndarray
        ] = {}

        missing_values = {
            "",
            "#N/A",
            "N/A",
            "NA",
            "NONE",
            "NAN",
            "NULL",
        }

        for drug, structure in smiles.items():

            if drug is None:
                continue

            drug = str(drug).strip()

            if not drug:
                continue

            if structure is None:
                continue

            structure = str(
                structure
            ).strip()

            if (
                not structure
                or structure.upper()
                in missing_values
            ):
                continue

            try:
                mol = Chem.MolFromSmiles(
                    structure
                )
            except Exception:
                mol = None

            if mol is None:
                continue

            try:

                vector = np.zeros(
                    (n_bits,),
                    dtype=np.uint8
                )

                DataStructs.ConvertToNumpyArray(
                    generator.GetFingerprint(mol),
                    vector
                )

                output[drug] = vector.astype(
                    np.float32,
                    copy=False
                )

            except Exception:
                continue

        return output

    # --------------------------------------------------------
    # RDKit unavailable / blocked
    # --------------------------------------------------------

    except Exception:

        from ML.feature_engineering.fingerprint_fallback import (
            make_fingerprints
        )

        output = make_fingerprints(
            smiles,
            n_bits=n_bits
        )

        # The fallback must also produce a dictionary.
        if not isinstance(output, dict):
            raise TypeError(
                "ML.feature_engineering.fingerprint_fallback.make_fingerprints() "
                "must return a dictionary."
            )

        return output


def _fingerprint_metadata() -> tuple[str, str]:
    """Return the fingerprint representation and backend version."""

    try:

        import rdkit

        return (
            "RDKit Morgan fingerprint (radius 2, 1024 bits)",
            rdkit.__version__,
        )

    except Exception as exc:

        return (
            "Pure-Python circular SMILES-graph approximation "
            "(radius 2, 1024 bits)",
            f"fallback after RDKit load failure: "
            f"{type(exc).__name__}",
        )


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(include_prism_drs: bool = False) -> dict:
    """Load local source files and construct reusable features."""

    aliases, smiles = _load_drug_maps()

    header = pd.read_csv(
        EXPRESSION_FILE,
        nrows=0,
        encoding="utf-8-sig"
    ).columns

    gene_candidates = {
        str(x).upper()
        for x in header
        if str(x).upper() not in {
            "DEPMAP_ID",
            "CELL_LINE_DISPLAY_NAME",
            "LINEAGE_1",
            "LINEAGE_2",
            "LINEAGE_3",
            "LINEAGE_4",
            "LINEAGE_5",
            "LINEAGE_6",
        }
    }

    vocab, ds = _read_gmt(
        gene_candidates,
        aliases
    )

    numeric_ds = _read_numeric_lincs(
        vocab,
        aliases
    )

    if (
        LINCS_WIDE_FILE.exists()
        and numeric_ds
    ):
        ds_source = (
            "LINCS Level 5 EH3226; "
            "24h/10uM; averaged across "
            "cell lines per drug"
        )

    elif numeric_ds:

        ds_source = (
            "LINCS numeric Level 5 CSV"
        )

    else:

        ds_source = (
            "signed LINCS GMT gene-list proxy"
        )

    if numeric_ds:
        ds = numeric_ds

    expression, cell_aliases, present = (
        _load_expression(vocab)
    )

    # Restrict all feature types to the
    # same measured-gene vocabulary.

    if present != vocab:

        remap = {
            gene: i
            for i, gene in enumerate(vocab)
        }

        keep = [
            remap[g]
            for g in present
        ]

        vocab = present

        ds = {
            drug: vector[keep]
            for drug, vector in ds.items()
        }

    gdsc_drs = _load_gdsc_resistance_signatures(
        vocab,
        aliases,
        expression,
        cell_aliases,
    )

    prism_drs: dict[str, np.ndarray] = {}
    prism_audit: dict = {
        "available": False,
        "signature_drugs": 0,
        "reason": "PRISM fallback was not requested",
    }
    drs = dict(gdsc_drs)
    prism_fallback_count = 0
    if include_prism_drs:
        prism_drs, prism_audit = _load_prism_resistance_signatures(
            aliases,
            expression,
        )
        prism_fallback = {
            drug: vector
            for drug, vector in prism_drs.items()
            if drug not in drs
        }
        drs.update(prism_fallback)
        prism_fallback_count = len(prism_fallback)

    if GDSC2_RESPONSE_FILE.exists() or GDSC1_RESPONSE_FILE.exists():
        drs_source = "official GDSC fitted dose-response tables; GDSC2 prioritized over GDSC1"
    else:
        drs_source = "project-local integrated GDSC columns in DrugComb table"
    if prism_fallback_count:
        drs_source += (
            f"; PRISM secondary-screen signatures as fallback for "
            f"{prism_fallback_count} drugs without GDSC signatures"
        )

    fingerprints = _make_fingerprints(
        smiles
    )

    structure_representation, structure_backend_version = (
        _fingerprint_metadata()
    )

    return {
        "aliases": aliases,

        "smiles": smiles,

        "fingerprints": fingerprints,

        "drug_signatures": ds,

        "resistance_signatures": drs,

        "gdsc_resistance_signatures": gdsc_drs,

        "prism_resistance_signatures": prism_drs,

        "drs_source_audit": {
            "gdsc_signature_drugs": len(gdsc_drs),
            "prism_signature_drugs": len(prism_drs),
            "prism_fallback_drugs": prism_fallback_count,
            "combined_signature_drugs": len(drs),
            "prism": prism_audit,
        },

        "drs_source": drs_source,

        "expression": expression,

        "cell_aliases": cell_aliases,

        "gene_vocab": vocab,

        "ds_source": ds_source,

        "structure_representation":
            structure_representation,

        "structure_backend_version":
            structure_backend_version,

        "drug_dim": 1024,

        "cell_dim": len(vocab),

        "drugs": sorted(
            set(fingerprints)
            | set(ds)
            | set(drs)
        ),

        "cell_lines": sorted(
            expression
        ),
    }


# ============================================================
# DRUG RESOLUTION
# ============================================================

def resolve_drug(
    name: object,
    context: dict
) -> str:

    key = normalize(name)

    return context[
        "aliases"
    ].get(
        key,
        key
    )


# ============================================================
# PAIR VECTOR
# ============================================================

def pair_vector(
    drug_a: str,
    drug_b: str,
    depmap_id: str,
    feature_set: str,
    context: dict,
    pair_schema: str = PAIR_SCHEMA_CONCAT,
) -> np.ndarray:

    """Create the exact numeric input vector used for fitting and serving."""

    if feature_set not in {
        "structure",
        "ds",
        "drs"
    }:
        raise ValueError(
            f"Unknown feature set: {feature_set}"
        )

    a = resolve_drug(
        drug_a,
        context
    )

    b = resolve_drug(
        drug_b,
        context
    )

    # Keep drug order deterministic.
    if b < a:
        a, b = b, a

    store = {
        "structure":
            context["fingerprints"],

        "ds":
            context["drug_signatures"],

        "drs":
            context["resistance_signatures"],
    }[feature_set]

    missing = [
        name
        for name in (a, b)
        if name not in store
    ]

    if missing:

        raise ValueError(
            f"No {feature_set} feature is available for: "
            f"{', '.join(missing)}"
        )

    cell = context[
        "expression"
    ].get(
        str(depmap_id)
    )

    if cell is None:

        cell_id = context[
            "cell_aliases"
        ].get(
            normalize(depmap_id)
        )

        cell = (
            context["expression"].get(
                cell_id
            )
            if cell_id
            else None
        )

    if cell is None:

        raise ValueError(
            "No baseline expression profile "
            f"is available for cell line: {depmap_id}"
        )

    vector_a = store[a]
    vector_b = store[b]
    if pair_schema == PAIR_SCHEMA_SYMMETRIC_SUM_ABSDIFF:
        if feature_set != "structure":
            raise ValueError(
                "The symmetric sum-and-difference schema is currently "
                "available for structure features only."
            )
        parts = [
            vector_a + vector_b,
            np.abs(vector_a - vector_b),
            cell,
        ]
    elif pair_schema == PAIR_SCHEMA_CONCAT:
        parts = [vector_a, vector_b, cell]
    else:
        raise ValueError(f"Unknown pair feature schema: {pair_schema}")

    return np.concatenate(parts).astype(np.float32, copy=False)


# ============================================================
# AUGMENT STRUCTURE FEATURES
# ============================================================

def augment_structure_features(
    frame: pd.DataFrame,
    context: dict
) -> None:

    """Recover valid structures from the matched training table.

    IMPORTANT:
    _make_fingerprints() now returns ONLY a dictionary,
    so dictionary.update() works correctly.
    """

    proposals: dict[
        str,
        set[str]
    ] = defaultdict(set)

    pairs = [
        ("drug_row", "smiles_row"),
        ("drug_col", "smiles_col"),
    ]

    for drug_col, smiles_col in pairs:

        if (
            drug_col not in frame.columns
            or smiles_col not in frame.columns
        ):
            continue

        unique_rows = (
            frame[
                [
                    drug_col,
                    smiles_col
                ]
            ]
            .dropna()
            .drop_duplicates()
        )

        for (
            drug_name,
            structure
        ) in unique_rows.itertuples(
            index=False,
            name=None
        ):

            key = resolve_drug(
                drug_name,
                context
            )

            structure = str(
                structure
            ).strip()

            if (
                key not in context["fingerprints"]
                and structure
            ):
                proposals[
                    key
                ].add(
                    structure
                )

    # Only accept drugs with exactly one
    # unique structure.

    unambiguous = {
        key: next(iter(values))
        for key, values in proposals.items()
        if len(values) == 1
    }

    if not unambiguous:
        return

    new_fingerprints = _make_fingerprints(
        unambiguous
    )

    if not isinstance(
        new_fingerprints,
        dict
    ):
        raise TypeError(
            "_make_fingerprints() must return "
            "a dictionary."
        )

    if new_fingerprints:

        context[
            "fingerprints"
        ].update(
            new_fingerprints
        )


# ============================================================
# FEATURE MATRIX
# ============================================================

def make_feature_matrix(
    frame: pd.DataFrame,
    feature_set: str,
    context: dict,
    pair_schema: str = PAIR_SCHEMA_CONCAT,
):
    """Return X, y, usable input rows, and transparent exclusion counts."""

    target = pd.to_numeric(
        frame["S"],
        errors="coerce"
    ).to_numpy(
        dtype=np.float64
    )

    if feature_set == "structure":

        augment_structure_features(
            frame,
            context
        )

    matrix: list[np.ndarray] = []

    kept: list[int] = []

    missing_drug = 0
    missing_cell = 0

    for index, row in enumerate(
        frame.itertuples(
            index=False
        )
    ):

        record = row._asdict()

        try:

            vector = pair_vector(
                record["drug_row"],
                record["drug_col"],
                record["depmap_id"],
                feature_set,
                context,
                pair_schema=pair_schema,
            )

        except ValueError as exc:

            message = str(exc).lower()

            if (
                "cell line" in message
                or "expression profile" in message
            ):
                missing_cell += 1

            else:
                missing_drug += 1

            continue

        if not np.isfinite(
            target[index]
        ):
            continue

        matrix.append(vector)
        kept.append(index)

    if not matrix:

        raise ValueError(
            f"No rows have complete "
            f"{feature_set} features and an S score."
        )

    counts = {
        "input_rows":
            len(frame),

        "usable_rows":
            len(kept),

        "missing_drug_feature":
            missing_drug,

        "missing_cell_expression":
            missing_cell,
    }

    return (
        np.vstack(matrix),
        target[
            np.asarray(kept)
        ],
        frame.iloc[
            kept
        ].reset_index(
            drop=True
        ),
        counts,
    )


def make_multimodal_feature_matrix(frame: pd.DataFrame, context: dict):
    """Align and concatenate rows with complete structure, DS, and DRS views.

    The output is the input for the level-2 stack. Keeping a shared set of
    row IDs makes each specialist's prediction refer to the same experiment.
    """
    feature_sets = ("structure", "ds", "drs")
    tagged = frame.copy()
    row_id_column = "multilevel_row_id"
    while row_id_column in tagged.columns:
        row_id_column = f"_{row_id_column}"
    tagged[row_id_column] = np.arange(len(tagged), dtype=np.int64)

    matrices: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    rows_by_view: dict[str, pd.DataFrame] = {}
    id_positions: dict[str, dict[int, int]] = {}
    view_counts: dict[str, int] = {}
    view_dimensions: list[int] = []
    for feature_set in feature_sets:
        X, y, used_rows, counts = make_feature_matrix(tagged, feature_set, context)
        row_ids = used_rows[row_id_column].to_numpy(dtype=np.int64)
        matrices[feature_set] = X
        targets[feature_set] = y
        rows_by_view[feature_set] = used_rows
        id_positions[feature_set] = {int(row_id): index for index, row_id in enumerate(row_ids)}
        view_counts[feature_set] = int(counts["usable_rows"])
        view_dimensions.append(int(X.shape[1]))

    reference_ids = rows_by_view["structure"][row_id_column].to_numpy(dtype=np.int64)
    complete_ids = [
        int(row_id) for row_id in reference_ids
        if all(int(row_id) in id_positions[feature_set] for feature_set in feature_sets[1:])
    ]
    if len(complete_ids) < 3:
        raise ValueError(
            "The multilevel stack needs at least three rows with complete structure, DS, and DRS features."
        )

    aligned_positions = {
        feature_set: np.asarray(
            [id_positions[feature_set][row_id] for row_id in complete_ids], dtype=np.int64
        )
        for feature_set in feature_sets
    }
    fused = np.concatenate(
        [matrices[feature_set][aligned_positions[feature_set]] for feature_set in feature_sets],
        axis=1,
    )
    structure_positions = aligned_positions["structure"]
    y = targets["structure"][structure_positions]
    used_rows = (
        rows_by_view["structure"]
        .iloc[structure_positions]
        .drop(columns=[row_id_column])
        .reset_index(drop=True)
    )
    exclusions = {
        "input_rows": int(len(frame)),
        "usable_rows": int(len(complete_ids)),
        "complete_case_rows": int(len(complete_ids)),
        "missing_any_view": int(len(frame) - len(complete_ids)),
        "view_coverage": view_counts,
        "view_dimensions": view_dimensions,
        "feature_rows": int(len(complete_ids)),
        "feature_columns": int(fused.shape[1]),
    }
    return fused, y, used_rows, exclusions
