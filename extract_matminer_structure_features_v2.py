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
from pymatgen.core import Composition, Structure

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent
STRUCTURES_DIR = PROJECT_DIR / "data" / "protocsp_generated_structures"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / "matminer_structure_features.csv"
TIMEOUT_SECONDS = 120
PROGRESS_STEP = 50


def handler(signum, frame):
    raise TimeoutError()


signal.signal(signal.SIGALRM, handler)


def extract_composition_from_filename(path: Path | str) -> str:
    stem = path.stem if isinstance(path, Path) else Path(path).stem
    match = re.match(r"^([^_]+)", stem)
    return match.group(1) if match else stem


def canonical_formula_from_filename(path: Path | str) -> str:
    raw = extract_composition_from_filename(path)
    try:
        return str(Composition(raw).reduced_formula)
    except Exception:
        return raw.replace(" ", "")


def load_target_canonical_formulas() -> set[str] | None:
    mat_path = RESULTS_DIR / "matminer_for_sisso_v2.csv"
    if not mat_path.exists():
        return None

    df = pd.read_csv(mat_path)
    if "canonical_formula" not in df.columns:
        return None

    formulas = set()
    for value in df["canonical_formula"].astype(str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none"}:
            continue
        try:
            formulas.add(str(Composition(text).reduced_formula))
        except Exception:
            formulas.add(text.replace(" ", ""))
    return formulas


def load_unique_structure_paths(structures_dir: Path, target_formulas: set[str] | None = None) -> dict[str, list[Path]]:
    cif_paths = sorted(structures_dir.glob("*.cif"))
    if not cif_paths:
        raise FileNotFoundError(f"No CIF files found in {structures_dir}")

    unique_paths: dict[str, list[Path]] = {}
    for cif_path in cif_paths:
        canonical = canonical_formula_from_filename(cif_path)
        if not canonical:
            continue
        if target_formulas is not None and canonical not in target_formulas:
            continue
        unique_paths.setdefault(canonical, []).append(cif_path)

    print(f"Found {len(cif_paths)} CIF files", flush=True)
    print(f"Loaded {len(unique_paths)} unique canonical compositions", flush=True)
    if target_formulas is not None:
        missing = sorted(target_formulas - set(unique_paths))
        print(
            f"Target matminer unique formulas: {len(target_formulas)}; "
            f"covered by CIF corpus: {len(unique_paths)}; "
            f"missing from CIF corpus: {len(missing)}",
            flush=True,
        )
        if missing:
            print("Sample missing formulas:", missing[:20], flush=True)
    return unique_paths


def build_featurizers() -> list:
    return [
        SiteStatsFingerprint.from_preset("CoordinationNumber_ward-prb-2017"),
        GlobalSymmetryFeatures(),
        DensityFeatures(),
    ]


def get_feature_names(featurizers: list) -> list[str]:
    names = ["composition", "canonical_formula"]
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
    for idx, (canonical, cif_path_list) in enumerate(items, start=1):
        if idx % PROGRESS_STEP == 0 or idx == total:
            print(f"Processed {idx}/{total} {description}...", flush=True)

        row_written = False
        for cif_path in cif_path_list:
            try:
                structure = Structure.from_file(str(cif_path))
                raw_composition = extract_composition_from_filename(cif_path)
                signal.alarm(TIMEOUT_SECONDS)
                try:
                    features = featurize_structure(structure, featurizers)
                finally:
                    signal.alarm(0)
                rows.append([raw_composition, canonical] + features)
                success += 1
                row_written = True
                break
            except TimeoutError:
                skipped += 1
                print(f"Skipped {canonical} from {cif_path.name}: timeout", flush=True)
            except Exception as exc:
                failed += 1
                print(f"Failed {canonical} from {cif_path.name}: {exc}", flush=True)

        if not row_written:
            print(f"No valid structure extracted for {canonical}", flush=True)

    return rows, success, skipped, failed


def print_summary(prefix: str, success: int, skipped: int, failed: int, df: pd.DataFrame) -> None:
    print(
        f"{prefix} summary: succeeded={success}, skipped={skipped}, failed={failed}, final shape={df.shape}",
        flush=True,
    )


def main() -> None:
    target_formulas = load_target_canonical_formulas()
    unique_paths = load_unique_structure_paths(STRUCTURES_DIR, target_formulas=target_formulas)
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
