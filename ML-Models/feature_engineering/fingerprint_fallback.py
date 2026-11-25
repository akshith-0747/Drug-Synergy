"""Deterministic pure-Python circular fingerprints for environments without RDKit.

When RDKit is available the main pipeline uses RDKit Morgan radius-2 bits. This
fallback parses the common SMILES graph grammar and hashes radius-0..2 atom
neighborhoods into a fixed bit vector. It is deliberately documented as an
approximation because it does not encode stereochemistry or RDKit atom invariants.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

_TWO_LETTER = {"Cl", "Br", "Si", "Na", "Ca", "Li", "Mg", "Al", "Fe", "Zn", "Cu", "Mn", "Co", "Ni", "Se", "As", "Ag", "Sn", "Hg", "Pt", "Au", "Pb", "Bi", "Cr", "Ba", "Sr", "Cs", "Rb", "Te"}


def _smiles_graph(smiles: str):
    atoms: list[str] = []
    aromatic: list[bool] = []
    charges: list[str] = []
    adjacency: list[list[tuple[int, str]]] = []
    current = None
    branches: list[int | None] = []
    rings: dict[str, tuple[int, str]] = {}
    bond = ""
    i = 0
    while i < len(smiles):
        char = smiles[i]
        if char in "-=#:/\\":
            bond = char
            i += 1
            continue
        if char == "(":
            branches.append(current)
            i += 1
            continue
        if char == ")":
            current = branches.pop() if branches else current
            i += 1
            continue
        if char == ".":
            current = None
            bond = ""
            i += 1
            continue
        if char.isdigit() or char == "%":
            if current is None:
                i += 1
                continue
            if char == "%" and i + 2 < len(smiles) and smiles[i + 1:i + 3].isdigit():
                ring_id, i = smiles[i + 1:i + 3], i + 3
            else:
                ring_id, i = char, i + 1
            if ring_id in rings:
                other, old_bond = rings.pop(ring_id)
                edge = bond or old_bond or "1"
                adjacency[current].append((other, edge))
                adjacency[other].append((current, edge))
            else:
                rings[ring_id] = (current, bond)
            bond = ""
            continue
        if char == "[":
            end = smiles.find("]", i + 1)
            if end < 0:
                i += 1
                continue
            body = smiles[i + 1:end]
            match = re.search(r"Cl|Br|[A-Z][a-z]?|[bcnops]", body)
            label = match.group(0) if match else "*"
            charge_match = re.search(r"[+-]+\d*", body)
            charge = charge_match.group(0) if charge_match else "0"
            is_aromatic = label.islower()
            i = end + 1
        else:
            if char == "*":
                label = "*"
                is_aromatic = False
                i += 1
            elif char.isalpha():
                token = smiles[i:i + 2]
                if token in _TWO_LETTER:
                    label = token
                    i += 2
                else:
                    label = char
                    i += 1
                is_aromatic = label.islower()
            else:
                i += 1
                continue
            charge = "0"
        atom_index = len(atoms)
        atoms.append(label)
        aromatic.append(is_aromatic)
        charges.append(charge)
        adjacency.append([])
        if current is not None:
            edge = bond or (":" if aromatic[current] and is_aromatic else "1")
            adjacency[current].append((atom_index, edge))
            adjacency[atom_index].append((current, edge))
        current = atom_index
        bond = ""
    return atoms, aromatic, charges, adjacency


def _stable_index(text: str, width: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % width


def circular_fingerprint(smiles: str, n_bits: int = 1024, radius: int = 2) -> np.ndarray:
    atoms, aromatic, charges, adjacency = _smiles_graph(smiles)
    bits = np.zeros(n_bits, dtype=np.float32)
    if not atoms:
        return bits
    labels = [f"{atom}|{len(neighbors)}|{int(aromatic[i])}|{charges[i]}" for i, (atom, neighbors) in enumerate(zip(atoms, adjacency))]
    for r in range(radius + 1):
        updated: list[str] = []
        for atom_index, label in enumerate(labels):
            neighborhood = sorted(f"{edge}:{labels[neighbor]}" for neighbor, edge in adjacency[atom_index])
            signature = f"{r}|{label}|" + "|".join(neighborhood)
            bits[_stable_index(signature, n_bits)] = 1.0
            updated.append(signature)
        labels = updated
    return bits


def make_fingerprints(smiles_by_drug: dict[str, str], n_bits: int = 1024) -> dict[str, np.ndarray]:
    output = {}
    for drug, smiles in smiles_by_drug.items():
        try:
            vector = circular_fingerprint(smiles, n_bits=n_bits)
        except Exception:
            continue
        if vector.any():
            output[drug] = vector
    return output
