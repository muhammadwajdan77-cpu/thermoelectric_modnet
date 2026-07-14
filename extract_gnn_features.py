import re
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure


CIF_DIR = Path("ProtoCSP/generated_structures")
RESULTS_DIR = Path("results")


def extract_composition_from_filename(filename: str) -> str:
    match = re.match(r"^([^_]+)", filename)
    return match.group(1) if match else filename


def load_unique_structures(cif_dir: Path) -> pd.Series:
    files = sorted(cif_dir.glob("*.cif"))
    if not files:
        raise FileNotFoundError(f"No CIF files found in {cif_dir}")

    unique_structures = {}
    skipped = 0
    for cif_path in files:
        composition = extract_composition_from_filename(cif_path.name)
        if composition in unique_structures:
            continue

        try:
            structure = Structure.from_file(str(cif_path))
            unique_structures[composition] = structure
        except Exception as exc:
            skipped += 1
            print(f"WARNING: Failed to load {cif_path.name}: {exc}")

    print(f"Loaded {len(unique_structures)} unique structures from {len(files)} CIF files")
    if skipped:
        print(f"Skipped {skipped} files due to load errors")
    return pd.Series(unique_structures)


def safe_get_features(featurizer_name: str, featurizer, structures: pd.Series) -> pd.DataFrame:
    try:
        features = featurizer.get_features(structures)
        if isinstance(features, pd.Series):
            features = features.to_frame().T
        elif isinstance(features, dict):
            features = pd.DataFrame(features)
        elif not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)
        return features
    except Exception as exc:
        raise RuntimeError(f"{featurizer_name} failed: {exc}") from exc


def save_features(df: pd.DataFrame, filename: Path) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filename, index=True)
    print(f"Saved {filename} with shape {df.shape}")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    structures = load_unique_structures(CIF_DIR)

    try:
        from mattervial import DescriptorMEGNetFeaturizer
        desc_ofm = DescriptorMEGNetFeaturizer(base_descriptor="l-OFM_v1")
        try:
            l_ofm_features = safe_get_features("l-OFM", desc_ofm, structures)
            save_features(l_ofm_features, RESULTS_DIR / "lOFM_features.csv")
        except RuntimeError as exc:
            print(f"WARNING: Skipping l-OFM feature extraction: {exc}")
    except Exception as exc:
        print(f"ERROR: Could not initialize DescriptorMEGNetFeaturizer: {exc}")

    try:
        from mattervial import MVLFeaturizer
        mvl = MVLFeaturizer()
        try:
            mvl_features = safe_get_features("MVL", mvl, structures)
            save_features(mvl_features, RESULTS_DIR / "MVL_features.csv")
        except RuntimeError as exc:
            print(f"WARNING: Skipping MVL feature extraction: {exc}")
    except Exception as exc:
        print(f"ERROR: Could not initialize MVLFeaturizer: {exc}")

    try:
        from mattervial import ORBFeaturizer
        orb = ORBFeaturizer(model_name="ORB_v3")
        try:
            orb_features = safe_get_features("ORB", orb, structures)
            save_features(orb_features, RESULTS_DIR / "ORB_features.csv")
        except RuntimeError as exc:
            print(f"WARNING: Skipping ORB feature extraction: {exc}")
    except Exception as exc:
        print(f"WARNING: ORBFeaturizer unavailable or failed to initialize: {exc}")
        print("Skipping ORB feature extraction.")

    print("Feature extraction complete.")
