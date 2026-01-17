from pathlib import Path
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split

# Species to genus mapping
SPECIES_TO_GENUS = {
    "Aedes aegypti": "Aedes",
    "Aedes albopictus": "Aedes",
    "Anopheles arabiensis": "Anopheles",
    "Anopheles gambiae": "Anopheles",
    "Culex pipiens": "Culex",
    "Culex quinquefasciatus": "Culex"
}

SEED = 42


def find_recording_id(wav_path, species_dir):
    """Find nearest parent directory starting with 'D_', or immediate parent as fallback."""
    current = wav_path.parent
    while True:
        if current.name.startswith('D_'):
            return current.name
        if current == species_dir:
            break
        if current.parent == current:  # reached root
            break
        current = current.parent
    
    # Fallback to immediate parent
    return wav_path.parent.name


def collect_recordings():
    """Scan Wingsbeats dataset and group files by recording_id."""
    base_path = Path("data/raw/Wingsbeats")
    
    if not base_path.exists():
        raise FileNotFoundError(f"Dataset not found: {base_path}")
    
    recordings = defaultdict(lambda: {"genus": None, "species": None, "files": []})
    
    # Iterate through species folders
    for species_dir in sorted(base_path.iterdir()):
        if not species_dir.is_dir():
            continue
        
        species_name = species_dir.name
        
        if species_name not in SPECIES_TO_GENUS:
            print(f"Warning: Unknown species folder '{species_name}', skipping")
            continue
        
        genus = SPECIES_TO_GENUS[species_name]
        
        # Find all wav files recursively
        for wav_file in species_dir.rglob("*.wav"):
            rec_id = find_recording_id(wav_file, species_dir)
            
            # Unique key: species::recording_id
            rec_key = f"{species_name}::{rec_id}"
            
            recordings[rec_key]["genus"] = genus
            recordings[rec_key]["species"] = species_name
            recordings[rec_key]["files"].append(wav_file)
    
    return recordings


def split_recordings(recordings):
    """Split recordings into train/val/test with stratification by genus."""
    # Sort for determinism
    rec_keys = sorted(recordings.keys())
    genera = [recordings[k]["genus"] for k in rec_keys]
    
    # Check if stratification is possible (need >= 2 samples per class)
    from collections import Counter
    genus_counts = Counter(genera)
    can_stratify = all(count >= 2 for count in genus_counts.values())
    
    if not can_stratify:
        print("Warning: Some genera have too few recordings for stratification, using random split")
        stratify_param = None
    else:
        stratify_param = genera
    
    # First split: 70% train, 30% temp (for val+test)
    train_recs, temp_recs, train_genera, temp_genera = train_test_split(
        rec_keys, genera,
        test_size=0.30,
        stratify=stratify_param,
        random_state=SEED
    )
    
    # Second split: check if temp can be stratified
    temp_genus_counts = Counter(temp_genera)
    can_stratify_temp = all(count >= 2 for count in temp_genus_counts.values())
    
    # Second split: split temp into 50/50 (15% val, 15% test of total)
    val_recs, test_recs, _, _ = train_test_split(
        temp_recs, temp_genera,
        test_size=0.50,
        stratify=temp_genera if can_stratify_temp else None,
        random_state=SEED
    )
    
    return train_recs, val_recs, test_recs


def create_split_dataframe(recording_keys, recordings):
    """Create DataFrame from recording keys, expanding to individual files."""
    repo_root = Path(__file__).resolve().parents[1]
    rows = []
    
    for rec_key in recording_keys:
        rec_data = recordings[rec_key]
        genus = rec_data["genus"]
        species = rec_data["species"]
        
        # Extract D_* folder as recording_id
        recording_id = rec_key.split("::", 1)[1]
        
        for wav_file in rec_data["files"]:
            # Relative path from repo root
            rel_path = wav_file.resolve().relative_to(repo_root)
            filepath = str(rel_path).replace("\\", "/")
            
            rows.append({
                "filepath": filepath,
                "genus": genus,
                "species": species,
                "recording_id": recording_id
            })
    
    return pd.DataFrame(rows)


def print_summary(train_df, val_df, test_df, recordings):
    """Print detailed split summary."""
    print("\n" + "="*90)
    print(" "*30 + "SPLIT SUMMARY")
    print("="*90)
    
    splits = {
        "Train": train_df,
        "Val": val_df,
        "Test": test_df
    }
    
    for split_name, df in splits.items():
        n_recordings = df["recording_id"].nunique()
        n_files = len(df)
        
        print(f"\n{split_name}:")
        print(f"  Total Recordings: {n_recordings:>6}")
        print(f"  Total Files:      {n_files:>6}")
        
        # Per-genus breakdown
        print(f"  Per Genus:")
        for genus in sorted(df["genus"].unique()):
            genus_df = df[df["genus"] == genus]
            n_rec = genus_df["recording_id"].nunique()
            n_files = len(genus_df)
            print(f"    {genus:<12} {n_rec:>4} recordings, {n_files:>6} files")
    
    print("\n" + "="*90)
    
    # Overall statistics
    total_recordings = len(recordings)
    total_files = sum(len(df) for df in splits.values())
    
    print(f"\nTotal Dataset:")
    print(f"  Recordings: {total_recordings}")
    print(f"  Files:      {total_files}")
    
    print("\nSplit Ratios:")
    for split_name, df in splits.items():
        ratio = (len(df) / total_files) * 100
        print(f"  {split_name:>5}: {ratio:>5.1f}%")
    
    print("="*90 + "\n")


def main():
    """Create train/val/test splits for Wingsbeats dataset."""
    
    print("\n" + "="*90)
    print(" "*25 + "WINGSBEATS DATASET SPLITTING")
    print("="*90)
    
    # Step 1: Collect all recordings
    print("\nScanning dataset...")
    recordings = collect_recordings()
    
    total_recs = len(recordings)
    total_files = sum(len(rec["files"]) for rec in recordings.values())
    
    print(f"Found {total_recs} recordings with {total_files} audio files")
    
    # Per-genus recording counts
    genus_counts = defaultdict(int)
    for rec in recordings.values():
        genus_counts[rec["genus"]] += 1
    
    print("\nRecordings per genus:")
    for genus in sorted(genus_counts.keys()):
        print(f"  {genus:<12} {genus_counts[genus]:>4} recordings")
    
    # Step 2: Split recordings
    print("\nSplitting at recording level (70% train, 15% val, 15% test)...")
    train_recs, val_recs, test_recs = split_recordings(recordings)
    
    # Step 3: Create DataFrames
    print("Creating split files...")
    train_df = create_split_dataframe(train_recs, recordings)
    val_df = create_split_dataframe(val_recs, recordings)
    test_df = create_split_dataframe(test_recs, recordings)
    
    # Step 4: Save CSVs
    output_dir = Path("data/splits")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / "wingsbeats_train.csv", index=False)
    val_df.to_csv(output_dir / "wingsbeats_val.csv", index=False)
    test_df.to_csv(output_dir / "wingsbeats_test.csv", index=False)
    
    print(f"\nSaved splits to {output_dir}/")
    print(f"  - wingsbeats_train.csv ({len(train_df)} files)")
    print(f"  - wingsbeats_val.csv ({len(val_df)} files)")
    print(f"  - wingsbeats_test.csv ({len(test_df)} files)")
    
    # Step 5: Print summary
    print_summary(train_df, val_df, test_df, recordings)
    
    print("Split creation complete!")


if __name__ == "__main__":
    main()


