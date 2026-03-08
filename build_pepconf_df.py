"""
build_pepconf_df.py
-------------------
Sample 25 systems from the pepconf database with >= 90 atoms, then
build a DataFrame with one row per (system x conformer x basis x method).

Total rows: 25 systems * 6 basis sets * 8 methods = 1200 rows
(conformer index is fixed at _0 for the system geometry used here)

Columns
-------
id                : str   – filename stem, e.g. "CREKA_0"
folder            : str   – sub-dataset name (bioactive, cyclic, …)
Geometry          : ndarray (N, 4) – [atomic_number, x, y, z] per atom (Angstrom)
qcel_mol          : qcel.models.Molecule
dimer_charge      : int
dimer_multiplicity: int
n_atoms           : int
basis             : str
method            : str
"""

import os
import re
import random

import numpy as np
import pandas as pd
import qcelemental as qcel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PEPCONF_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDERS = ["bioactive", "cyclic", "dipeptide", "disulfide", "tripeptide"]

BASIS_SETS = [
    "cc-pVDZ",
    "cc-pVTZ",
    "cc-pVQZ",
    "aug-cc-pVDZ",
    "aug-cc-pVTZ",
    "aug-cc-pVQZ",
]

N_SAMPLE = 25
MIN_ATOMS = 90
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Helper: element symbol -> atomic number
# ---------------------------------------------------------------------------
def _symbol_to_z(sym: str) -> int:
    return qcel.periodictable.to_Z(sym.capitalize())


# ---------------------------------------------------------------------------
# Parse statistics.txt to collect (folder, system_id, n_atoms)
# Only the _0 conformer is used as the canonical geometry.
# ---------------------------------------------------------------------------
def collect_candidates() -> list[tuple[str, str, int]]:
    """Return list of (folder, system_id_stem, n_atoms) for n_atoms >= MIN_ATOMS."""
    candidates = []
    for folder in FOLDERS:
        stats_path = os.path.join(PEPCONF_DIR, folder, "xyz", "statistics.txt")
        text = open(stats_path).read().strip().splitlines()
        i = 0
        while i < len(text):
            line = text[i].strip()
            if line.startswith("==>"):
                fname = line.split("==> ")[1].split(" <==")[0]
                n_atoms = int(text[i + 1].strip())
                i += 2
                # Only keep _0 conformers as canonical representatives
                if fname.endswith("_0.xyz") and n_atoms >= MIN_ATOMS:
                    stem = fname.replace(".xyz", "")
                    candidates.append((folder, stem, n_atoms))
            else:
                i += 1
    return candidates


# ---------------------------------------------------------------------------
# Parse a single xyz file
# ---------------------------------------------------------------------------
def parse_xyz(path: str) -> tuple[int, int, np.ndarray]:
    """
    Parse a pepconf xyz file.

    Returns
    -------
    charge : int
    multiplicity : int
    geometry : ndarray shape (N, 4) — columns [atomic_number, x, y, z]
    """
    lines = open(path).read().strip().splitlines()
    # line 0: atom count (ignored – we count directly)
    # line 1: charge multiplicity
    charge, mult = (int(v) for v in lines[1].split())
    rows = []
    for line in lines[2:]:
        parts = line.split()
        if not parts:
            continue
        sym = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        rows.append([_symbol_to_z(sym), x, y, z])
    geometry = np.array(rows, dtype=float)
    return charge, mult, geometry


# ---------------------------------------------------------------------------
# Build molecule with qcelemental
# ---------------------------------------------------------------------------
def xyz_to_mol(xyz_text: str) -> qcel.models.Molecule:
    return qcel.models.Molecule.from_data(xyz_text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Sample 25 systems reproducibly ---
    candidates = collect_candidates()
    print(f"Candidate systems with >= {MIN_ATOMS} atoms: {len(candidates)}")

    rng = random.Random(RANDOM_SEED)
    sampled = rng.sample(candidates, N_SAMPLE)
    sampled.sort(key=lambda t: (t[0], t[1]))  # deterministic order after sampling

    print(f"Sampled {len(sampled)} systems:")
    for folder, stem, n in sampled:
        print(f"  [{folder}] {stem}  ({n} atoms)")

    # --- Build rows ---
    records = []
    for folder, stem, n_atoms in sampled:
        xyz_path = os.path.join(PEPCONF_DIR, folder, "xyz", f"{stem}.xyz")
        xyz_text = open(xyz_path).read()

        charge, mult, geometry = parse_xyz(xyz_path)
        mol = xyz_to_mol(xyz_text)

        for basis in BASIS_SETS:
                records.append(
                    {
                        "id": stem,
                        "folder": folder,
                        "Geometry": geometry,
                        "qcel_mol": mol,
                        "dimer_charge": charge,
                        "dimer_multiplicity": mult,
                        "n_atoms": n_atoms,
                        "basis": basis,
                    }
                )

    df = pd.DataFrame(records)
    expected_rows = N_SAMPLE * len(BASIS_SETS)
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"
    print(f"\nDataFrame shape: {df.shape}")
    print(df[["id", "folder", "n_atoms", "basis"]].head(20).to_string())

    # --- Save ---
    out_path = os.path.join(PEPCONF_DIR, "pepconf_sampled.pkl")
    df.to_pickle(out_path)
    print(f"\nSaved DataFrame to {out_path}")

    return df


if __name__ == "__main__":
    df = main()
