"""Extract MODNet MatminerAll2023 structure and site features from CIF files.

This script loads one unique structure per composition from
ProtoCSP/generated_structures/*.cif, computes MODNet's rich structure and site
features, and saves them to a separate full-feature output.

It first tests the pipeline on 5 structures, prints the results, and then
processes the full unique structure set.
"""

import json
import os
import re
import signal
import warnings
from pathlib import Path

import pandas as pd
from modnet.featurizers.presets import MatminerAll2023Featurizer
from pymatgen.core import Composition, Structure

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent
STRUCTURES_DIR = PROJECT_DIR / "data" / "protocsp_generated_structures"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / "matminer_structure_features_full.csv"
CHECKPOINT_PATH = RESULTS_DIR / "matminer_structure_features_full_partial.jsonl"
TIMEOUT_SECONDS = 300
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


def extract_rank_from_filename(path: Path | str) -> int | None:
    stem = path.stem if isinstance(path, Path) else Path(path).stem
    match = re.search(r"_rank_(\d+)$", stem)
    return int(match.group(1)) if match else None


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


def sort_paths_for_canonical(paths: list[Path]) -> list[Path]:
    # Deterministic tie-break for duplicate rank-1 files: prefer the lexicographically
    # smallest filename for each canonical formula. The caller then tries paths in
    # this order until a valid structure parses, ensuring reproducible output.
    return sorted(paths, key=lambda p: p.name)


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

        rank = extract_rank_from_filename(cif_path)
        if rank is not None and rank != 1:
            continue

        unique_paths.setdefault(canonical, []).append(cif_path)

    unique_paths = {canonical: sort_paths_for_canonical(paths) for canonical, paths in unique_paths.items()}

    printed_rank1_paths = sum(len(paths) for paths in unique_paths.values())
    print(f"Found {len(cif_paths)} CIF files", flush=True)
    print(f"Filtered to rank-1 candidates: {printed_rank1_paths} CIF files", flush=True)
    print(f"Loaded {len(unique_paths)} unique canonical compositions from rank-1 candidates", flush=True)
    if target_formulas is not None:
        missing = sorted(target_formulas - set(unique_paths))
        print(
            f"Target matminer unique formulas: {len(target_formulas)}; "
            f"covered by rank-1 CIF corpus: {len(unique_paths)}; "
            f"missing from rank-1 CIF corpus: {len(missing)}",
            flush=True,
        )
        if missing:
            print("Sample missing formulas:", missing[:20], flush=True)
    return unique_paths


def build_featurizer() -> MatminerAll2023Featurizer:
    return MatminerAll2023Featurizer(impute_nan=True)


def featurize_structure(
    structure: Structure, featurizer: MatminerAll2023Featurizer
) -> dict[str, object]:
    frame = pd.DataFrame({"structure": [structure]})
    structure_features = featurizer.featurize_structure(frame)
    site_features = featurizer.featurize_site(frame)
    structure_features = structure_features.drop(columns=["structure"], errors="ignore")
    site_features = site_features.drop(columns=["Input data|structure"], errors="ignore")
    feature_frame = pd.concat(
        [
            structure_features,
            site_features,
        ],
        axis=1,
    )
    feature_frame = feature_frame.loc[:, ~feature_frame.columns.duplicated()]
    return feature_frame.iloc[0].to_dict()


def load_checkpoint(path: Path) -> list[tuple[str, str, dict[str, object]]]:
    rows: list[tuple[str, str, dict[str, object]]] = []
    if not path.exists():
        return rows

    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            canonical = record.get("canonical_formula")
            if not canonical or canonical in seen:
                continue
            rows.append(
                (
                    record["composition"],
                    canonical,
                    record["features"],
                )
            )
            seen.add(canonical)
    return rows


def append_checkpoint(
    path: Path,
    raw_composition: str,
    canonical: str,
    features: dict[str, object],
) -> None:
    record = {
        "composition": raw_composition,
        "canonical_formula": canonical,
        "features": features,
    }
    with path.open("a", encoding="utf-8") as handle:
        json.dump(
            record,
            handle,
            ensure_ascii=True,
            default=lambda value: value.item() if hasattr(value, "item") else str(value),
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def process_paths(
    paths: dict[str, Path],
    featurizer: MatminerAll2023Featurizer,
    max_items: int | None = None,
    description: str = "structures",
    checkpoint_path: Path | None = None,
) -> tuple[list[tuple[str, str, dict[str, object]]], int, int, int]:
    rows: list[tuple[str, str, dict[str, object]]] = []
    success = 0
    skipped = 0
    failed = 0
    total = len(paths) if max_items is None else min(max_items, len(paths))

    items = list(paths.items())[:total]
    for idx, (canonical, cif_path_list) in enumerate(items, start=1):
        if idx % PROGRESS_STEP == 0 or idx == total:
            print(f"Processed {idx}/{total} {description}...", flush=True)

        row_written = False
        ordered_paths = sort_paths_for_canonical(cif_path_list)
        for cif_path in ordered_paths:
            try:
                structure = Structure.from_file(str(cif_path))
                raw_composition = extract_composition_from_filename(cif_path)
                signal.alarm(TIMEOUT_SECONDS)
                try:
                    features = featurize_structure(structure, featurizer)
                finally:
                    signal.alarm(0)
                rows.append((raw_composition, canonical, features))
                if checkpoint_path is not None:
                    append_checkpoint(checkpoint_path, raw_composition, canonical, features)
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


def rows_to_dataframe(rows: list[tuple[str, str, dict[str, object]]]) -> pd.DataFrame:
    feature_names = sorted({name for _, _, features in rows for name in features})
    data = [
        [raw_composition, canonical] + [features.get(name) for name in feature_names]
        for raw_composition, canonical, features in rows
    ]
    return pd.DataFrame(
        data,
        columns=["composition", "canonical_formula"] + feature_names,
    )


def print_summary(prefix: str, success: int, skipped: int, failed: int, df: pd.DataFrame) -> None:
    print(
        f"{prefix} summary: succeeded={success}, skipped={skipped}, failed={failed}, final shape={df.shape}",
        flush=True,
    )


def main() -> None:
    target_formulas = load_target_canonical_formulas()
    unique_paths = load_unique_structure_paths(STRUCTURES_DIR, target_formulas=target_formulas)
    featurizer = build_featurizer()
    checkpoint_rows = load_checkpoint(CHECKPOINT_PATH)
    completed = {canonical for _, canonical, _ in checkpoint_rows}
    if checkpoint_rows:
        print(
            f"Resuming from checkpoint: {len(checkpoint_rows)} completed structures",
            flush=True,
        )

    if os.environ.get("SKIP_SMOKE_TEST") == "1" or checkpoint_rows:
        test_rows, test_success, test_skipped, test_failed = [], 0, 0, 0
        test_df = pd.DataFrame(columns=["composition", "canonical_formula"])
        print("Skipping smoke test for resumed run.", flush=True)
    else:
        print("\nTesting on first 5 unique structures...", flush=True)
        test_rows, test_success, test_skipped, test_failed = process_paths(
            unique_paths, featurizer, max_items=5, description="test structures"
        )

        if test_rows:
            test_df = rows_to_dataframe(test_rows)
            print(test_df.to_string(index=False), flush=True)
        else:
            test_df = pd.DataFrame(columns=["composition", "canonical_formula"])
            print("No test rows were generated.", flush=True)

    print_summary("Test", test_success, test_skipped, test_failed, test_df)

    print("\nProcessing full set of unique structures...", flush=True)
    remaining_paths = {
        canonical: paths
        for canonical, paths in unique_paths.items()
        if canonical not in completed
    }
    full_limit = os.environ.get("FULL_MAX_ITEMS")
    rows, success, skipped, failed = process_paths(
        remaining_paths,
        featurizer,
        max_items=int(full_limit) if full_limit else None,
        description="structures",
        checkpoint_path=CHECKPOINT_PATH,
    )
    df = rows_to_dataframe(checkpoint_rows + rows)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved structure-based Matminer features: {OUTPUT_PATH}", flush=True)
    print_summary("Full run", success, skipped, failed, df)


if __name__ == "__main__":
    main()
