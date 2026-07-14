"""
Fast ProtoCSP Index Builder using pickle + multiprocessing
Saves to lemat_formula_indexed.pkl
"""
import os, sys, gc, pickle, glob, csv
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd

CSV_DIR = "lemat_unique_csv_500_parts"
OUTPUT  = "lemat_formula_indexed.pkl"
WORKERS = min(8, cpu_count())

def process_file(file_path):
    """Process one CSV file, return dict of anon_formula -> list of entries."""
    from pymatgen.core import Structure, Lattice
    local_index = defaultdict(list)
    try:
        chunks = pd.read_csv(file_path, on_bad_lines='warn',
                             quoting=csv.QUOTE_MINIMAL, chunksize=2000)
        for chunk in chunks:
            for _, row in chunk.iterrows():
                try:
                    # Try to reconstruct structure from row
                    if 'structure' in row and pd.notna(row.get('structure')):
                        import json
                        s_dict = json.loads(row['structure']) if isinstance(row['structure'], str) else row['structure']
                        struct = Structure.from_dict(s_dict)
                    elif all(c in row for c in ['a','b','c','alpha','beta','gamma']):
                        lattice = Lattice.from_parameters(
                            float(row['a']), float(row['b']), float(row['c']),
                            float(row['alpha']), float(row['beta']), float(row['gamma'])
                        )
                        species = str(row.get('species', row.get('elements', ''))).split()
                        coords  = []
                        for i in range(len(species)):
                            x = float(row.get(f'x{i}', row.get(f'frac_x{i}', 0)))
                            y = float(row.get(f'y{i}', row.get(f'frac_y{i}', 0)))
                            z = float(row.get(f'z{i}', row.get(f'frac_z{i}', 0)))
                            coords.append([x, y, z])
                        if not coords: continue
                        struct = Structure(lattice, species, coords)
                    else:
                        continue

                    anon = struct.composition.anonymized_formula
                    entry = {
                        'structure': struct,
                        'formula': str(struct.composition.reduced_formula),
                        'space_group': None,
                        'source': os.path.basename(file_path)
                    }
                    # Add any extra columns
                    for col in ['spacegroup_number','energy_per_atom','band_gap']:
                        if col in row and pd.notna(row[col]):
                            entry[col] = row[col]
                    local_index[anon].append(entry)
                except Exception:
                    continue
    except Exception as e:
        print(f"  Skipping {os.path.basename(file_path)}: {e}")
    return dict(local_index)


def merge_indices(results):
    merged = defaultdict(list)
    for r in results:
        for k, v in r.items():
            merged[k].extend(v)
    return dict(merged)


def main():
    files = sorted(glob.glob(os.path.join(CSV_DIR, "part_*.csv")))
    print(f"Found {len(files)} CSV files")
    print(f"Using {WORKERS} workers")

    # Check existing pkl
    if os.path.exists(OUTPUT):
        print(f"Index already exists: {OUTPUT}")
        with open(OUTPUT, 'rb') as f:
            idx = pickle.load(f)
        print(f"Loaded: {len(idx)} unique stoichiometries")
        return

    # Process in batches to save memory
    BATCH = 50
    all_results = []

    for i in range(0, len(files), BATCH):
        batch = files[i:i+BATCH]
        print(f"\nBatch {i//BATCH + 1}/{(len(files)+BATCH-1)//BATCH} ({len(batch)} files)")
        with Pool(WORKERS) as pool:
            results = list(tqdm(pool.imap(process_file, batch), total=len(batch)))
        all_results.extend(results)
        gc.collect()

    print("\nMerging all results...")
    final_index = merge_indices(all_results)

    total = sum(len(v) for v in final_index.values())
    print(f"Total: {len(final_index)} stoichiometries, {total} structures")

    print(f"Saving to {OUTPUT}...")
    with open(OUTPUT, 'wb') as f:
        pickle.dump(final_index, f, protocol=pickle.HIGHEST_PROTOCOL)

    size = os.path.getsize(OUTPUT) / (1024**2)
    print(f"Saved! Size: {size:.1f} MB")


if __name__ == "__main__":
    main()
