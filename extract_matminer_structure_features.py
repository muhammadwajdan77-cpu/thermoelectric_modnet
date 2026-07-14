"""Extract structure-based Matminer features from crystal CIF files.

This script loads one unique structure per composition from
ProtoCSP/generated_structures/*.cif, computes true structure-based features
with Matminer, and saves them to results/matminer_structure_features.csv.
"""

import re
import time
import warnings
from pathlib import Path

import pandas as pd
from matminer.featurizers.structure import (
    SiteStatsFingerprint,
    StructuralComplexity,
    GlobalSymmetryFeatures,
    DensityFeatures,
    MaximumPackingEfficiency,
)
from pymatgen.core import Structure

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent
STRUCTURES_DIR = PROJECT_DIR / "ProtoCSP" / "generated_structures"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_composition_from_filename(path: Path) -> str:
    stem = path.stem
    match = re.match(r"^(.+?)_doping_", stem)
    if match:
        return match.group(1)
    return stem


def load_unique_structure_paths(structures_dir: Path) -> dict[str, Path]:
    cif_paths = sorted(structures_dir.glob("*.cif"))
    if not cif_paths:
        raise FileNotFoundError(f"No CIF files found in {structures_dir}")

    unique_paths: dict[str, Path] = {}
    for cif_path in cif_paths:
        composition = extract_composition_from_filename(cif_path)
        if composition not in unique_paths:
            unique_paths[composition] = cif_path

    print(f"Found {len(cif_paths)} CIF files", flush=True)
    print(f"Loaded {len(unique_paths)} unique compositions", flush=True)
    return unique_paths


def build_featurizers() -> list:
    return [
        SiteStatsFingerprint.from_preset("CoordinationNumber_ward-prb-2017"),
        StructuralComplexity(),
        GlobalSymmetryFeatures(),
        DensityFeatures(),
        MaximumPackingEfficiency(),
    ]


def main() -> None:
    unique_paths = load_unique_structure_paths(STRUCTURES_DIR)
    featurizers = build_featurizers()

    feature_names = ["composition"]
    for f in featurizers:
        feature_names.extend(f.feature_labels())

    rows = []
    success = 0
    failed = 0
    total = len(unique_paths)
    start_time = time.perf_counter()

    for idx, (composition, cif_path) in enumerate(unique_paths.items(), start=1):
        structure_start = time.perf_counter()
        try:
            structure = Structure.from_file(str(cif_path))
            row = [composition]
            for featurizer in featurizers:
                row.extend(featurizer.featurize(structure))
            rows.append(row)
            success += 1
        except Exception as exc:
            failed += 1
            print(f"Failed {composition} ({cif_path.name}): {exc}", flush=True)

        structure_elapsed = time.perf_counter() - structure_start
        if idx % 50 == 0 or idx == total:
            elapsed = time.perf_counter() - start_time
            avg = elapsed / idx
            remaining = avg * (total - idx)
            print(
                f"  Progress: {idx}/{total} | success={success} | failed={failed} "
                f"| last={structure_elapsed:.1f}s | avg={avg:.1f}s | ETA={remaining/60:.1f}m",
                flush=True,
            )

    df = pd.DataFrame(rows, columns=feature_names)
    output_path = RESULTS_DIR / "matminer_structure_features.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved structure-based Matminer features: {output_path}", flush=True)
    print(f"Output shape: {df.shape}", flush=True)
    print(f"Succeeded: {success} | Failed: {failed}", flush=True)

    existing_path = RESULTS_DIR / "matminer_for_sisso.csv"
    if existing_path.exists():
        existing_cols = pd.read_csv(existing_path, nrows=0).columns.tolist()
        existing_cols = [c for c in existing_cols if c != "target"]
        output_cols = [c for c in df.columns if c != "composition"]

        shared = sorted(set(output_cols) & set(existing_cols))
        new_only = sorted(set(output_cols) - set(existing_cols))

        print("\nExisting composition-only Matminer columns:")
        print(f"  {len(existing_cols)} columns loaded from {existing_path.name}")
        print("\nStructure-based output columns:")
        print(f"  {len(output_cols)} columns")
        print(", ".join(output_cols))

        print("\nColumns unique to structure-based features:")
        if new_only:
            print(f"  {len(new_only)} distinct columns")
            print(", ".join(new_only))
        else:
            print("  None")

        print("\nColumns shared with composition-only Matminer:")
        if shared:
            print(f"  {len(shared)} shared columns")
            print(", ".join(shared))
        else:
            print("  None")
    else:
        print(f"Warning: existing file {existing_path} not found; could not verify column overlap.")


if __name__ == "__main__":
    main()
