"""Extract fast structure-based Matminer features from CIF files.

This script loads one unique structure per composition from
ProtoCSP/generated_structures/*.cif, computes structure-only features with
Matminer, and saves them to results/matminer_structure_features.csv.

It first tests the pipeline on 5 structures, prints the results, and then
processes the full unique structure set.
"""

import re
import signal
import warnings
from pathlib import Path

import pandas as pd
from matminer.featurizers.structure import (
    DensityFeatures,
    GlobalSymmetryFeatures,
    SiteStatsFingerprint,
)
from pymatgen.core import Structure

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent
STRUCTURES_DIR = PROJECT_DIR / "ProtoCSP" / "generated_structures"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / "matminer_structure_features.csv"
TIMEOUT_SECONDS = 30
PROGRESS_STEP = 50


def handler(signum, frame):
    raise TimeoutError()


signal.signal(signal.SIGALRM, handler)


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
        GlobalSymmetryFeatures(),
        DensityFeatures(),
    ]


def get_feature_names(featurizers: list) -> list[str]:
    names = ["composition"]
    for featurizer in featurizers:
        names.extend(featurizer.feature_labels())
    return names


def featurize_structure(structure: Structure, featurizers: list) -> list:
    row: list = []
    for featurizer in featurizers:
        row.extend(featurizer.featurize(structure))
    return row


def process_paths(
    paths: dict[str, Path],
    featurizers: list,
    max_items: int | None = None,
    description: str = "structures",
) -> tuple[list[list], int, int, int]:
    rows: list[list] = []
    success = 0
    skipped = 0
    failed = 0
    total = len(paths) if max_items is None else min(max_items, len(paths))

    items = list(paths.items())[:total]
    for idx, (composition, cif_path) in enumerate(items, start=1):
        if idx % PROGRESS_STEP == 0 or idx == total:
            print(f"Processed {idx}/{total} {description}...", flush=True)

        try:
            structure = Structure.from_file(str(cif_path))
            signal.alarm(TIMEOUT_SECONDS)
            try:
                features = featurize_structure(structure, featurizers)
            finally:
                signal.alarm(0)
            rows.append([composition] + features)
            success += 1
        except TimeoutError:
            skipped += 1
            print(f"Skipped {composition}: timeout", flush=True)
        except Exception as exc:
            failed += 1
            print(f"Failed {composition} ({cif_path.name}): {exc}", flush=True)

    return rows, success, skipped, failed


def print_summary(prefix: str, success: int, skipped: int, failed: int, df: pd.DataFrame) -> None:
    print(
        f"{prefix} summary: succeeded={success}, skipped={skipped}, failed={failed}, final shape={df.shape}",
        flush=True,
    )


def main() -> None:
    unique_paths = load_unique_structure_paths(STRUCTURES_DIR)
    featurizers = build_featurizers()
    feature_names = get_feature_names(featurizers)

    print("\nTesting on first 5 unique structures...", flush=True)
    test_rows, test_success, test_skipped, test_failed = process_paths(
        unique_paths, featurizers, max_items=5, description="test structures"
    )

    if test_rows:
        test_df = pd.DataFrame(test_rows, columns=feature_names)
        print(test_df.to_string(index=False), flush=True)
    else:
        print("No test rows were generated.", flush=True)

    print_summary("Test", test_success, test_skipped, test_failed, test_df if test_rows else pd.DataFrame(columns=feature_names))

    print("\nProcessing full set of unique structures...", flush=True)
    rows, success, skipped, failed = process_paths(unique_paths, featurizers, description="structures")
    df = pd.DataFrame(rows, columns=feature_names)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved structure-based Matminer features: {OUTPUT_PATH}", flush=True)
    print_summary("Full run", success, skipped, failed, df)


if __name__ == "__main__":
    main()
