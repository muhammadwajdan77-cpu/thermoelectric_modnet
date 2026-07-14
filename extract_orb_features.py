#!/usr/bin/env python3
"""
Extract ORB features using MatterVial ORBFeaturizer for unique CIF structures.
Output: results/ORB_features.csv
"""

import sys
import glob
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure

# MatterVial imports
try:
    from mattervial.featurizers import ORBFeaturizer
except ImportError:
    try:
        from mattervial.featurizer import ORBFeaturizer
    except ImportError:
        print("ERROR: mattervial not installed in current environment")
        sys.exit(1)


def extract_composition_from_filename(filename: str) -> str:
    return filename.split("_", 1)[0]


def load_unique_structures(cif_dir: Path) -> dict[str, Structure]:
    cif_files = sorted(cif_dir.glob("*.cif"))
    if not cif_files:
        raise FileNotFoundError(f"No CIF files found in {cif_dir}")

    unique_structures: dict[str, Structure] = {}
    skipped = 0
    for cif_path in cif_files:
        composition = extract_composition_from_filename(cif_path.name)
        if composition in unique_structures:
            continue

        try:
            unique_structures[composition] = Structure.from_file(str(cif_path))
        except Exception as exc:
            skipped += 1
            print(f"WARNING: Failed to load {cif_path.name}: {exc}")

    print(f"Loaded {len(unique_structures)} unique structures from {len(cif_files)} CIF files")
    if skipped:
        print(f"Skipped {skipped} files due to load errors")
    return unique_structures


def normalize_features(features):
    if isinstance(features, pd.Series):
        return features.to_frame().T
    if isinstance(features, dict):
        return pd.DataFrame([features])
    if isinstance(features, pd.DataFrame):
        return features
    return pd.DataFrame(features)


def main() -> None:
    print("=" * 60)
    print("ORB FEATURE EXTRACTION")
    print("=" * 60)

    cif_dir = Path("ProtoCSP/generated_structures")
    output_file = Path("results") / "ORB_features.csv"

    if not cif_dir.exists():
        print(f"ERROR: CIF directory not found: {cif_dir}")
        sys.exit(1)

    print(f"\nSTEP 1 - LOAD UNIQUE STRUCTURES")
    print("=" * 60)
    unique_structures = load_unique_structures(cif_dir)
    if not unique_structures:
        print("ERROR: No unique structures were loaded")
        sys.exit(1)

    structures = pd.Series(list(unique_structures.values()))
    compositions = list(unique_structures.keys())
    formulas = [structure.composition.reduced_formula for structure in structures]

    print(f"\nSTEP 2 - INITIALIZE ORBFeaturizer")
    print("=" * 60)
    try:
        orb = ORBFeaturizer(model_name="ORB_v3")
    except Exception as exc:
        print(f"ERROR: Failed to initialize ORBFeaturizer: {exc}")
        sys.exit(1)

    print("ORBFeaturizer initialized successfully")

    print(f"\nSTEP 3 - EXTRACT FEATURES")
    print("=" * 60)
    try:
        features = orb.get_features(structures)
        features_df = normalize_features(features)
    except Exception as exc:
        print(f"ERROR: ORB feature extraction failed: {exc}")
        sys.exit(1)

    if features_df.shape[0] != len(structures):
        print(
            f"WARNING: Extracted {features_df.shape[0]} rows, but expected {len(structures)} unique structures"
        )

    print(f"ORB features shape: {features_df.shape}")

    print(f"\nSTEP 4 - BUILD OUTPUT")
    print("=" * 60)
    results_df = features_df.copy()
    results_df.insert(0, "composition", compositions)
    results_df.insert(1, "pretty_formula", formulas)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)

    print(f"Saved ORB features to: {output_file}")
    print(f"Rows: {results_df.shape[0]}")
    print(f"Columns: {results_df.shape[1]}")
    print(f"Feature columns: {results_df.shape[1] - 2}")

    print(f"\nSTEP 5 - SUMMARY")
    print("=" * 60)
    print(results_df.head())
    print(f"\nFEATURE EXTRACTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
