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


def collect_audio_files():
    """Scan Abuzz dataset and collect all wav files."""
    base_path = Path("data/raw/Abuzz")
    repo_root = Path(__file__).resolve().parents[1]
    
    if not base_path.exists():
        raise FileNotFoundError(f"Dataset not found: {base_path}")
    
    files_data = []
    
    # Iterate through species folders
    for species_dir in sorted(base_path.iterdir()):
        if not species_dir.is_dir():
            continue
        
        species_name = species_dir.name
        
        if species_name not in SPECIES_TO_GENUS:
            print(f"Warning: Unknown species folder '{species_name}', skipping")
            continue
        
        genus = SPECIES_TO_GENUS[species_name]
        
        # Flat structure - glob for wav files
        for wav_file in species_dir.glob("*.wav"):
            # Relative path from repo root
            rel_path = wav_file.resolve().relative_to(repo_root)
            filepath = str(rel_path).replace("\\", "/")
            
            # Unique recording_id for schema consistency
            recording_id = f"{species_name}::{wav_file.name}"
            
            files_data.append({
                "filepath": filepath,
                "genus": genus,
                "species": species_name,
                "recording_id": recording_id
            })
    
    return files_data


def split_data(files_data):
    """Split data into train/val/test with stratification by species."""
    # Sort for determinism
    files_data = sorted(files_data, key=lambda x: x["filepath"])
    
    # Extract species for stratification
    species = [f["species"] for f in files_data]
    
    # Check if stratification is possible (need >= 2 samples per class)
    from collections import Counter
    species_counts = Counter(species)
    can_stratify = all(count >= 2 for count in species_counts.values())
    
    if not can_stratify:
        print("Warning: Some species have too few files for stratification, using random split")
        stratify_param = None
    else:
        stratify_param = species
    
    # First split: 70% train, 30% temp (for val+test)
    train_data, temp_data, train_species, temp_species = train_test_split(
        files_data, species,
        test_size=0.30,
        stratify=stratify_param,
        random_state=SEED
    )
    
    # Second split: check if temp can be stratified
    temp_species_counts = Counter(temp_species)
    can_stratify_temp = all(count >= 2 for count in temp_species_counts.values())
    
    # Second split: split temp into 50/50 (15% val, 15% test of total)
    val_data, test_data, _, _ = train_test_split(
        temp_data, temp_species,
        test_size=0.50,
        stratify=temp_species if can_stratify_temp else None,
        random_state=SEED
    )
    
    return train_data, val_data, test_data


def print_summary(train_df, val_df, test_df):
    """Print detailed split summary."""
    print("\n" + "="*90)
    print(" "*30 + "SPLIT SUMMARY")
    print("="*90)
    
    splits = {
        "Train": train_df,
        "Val": val_df,
        "Test": test_df
    }
    
    total_files = sum(len(df) for df in splits.values())
    
    for split_name, df in splits.items():
        n_files = len(df)
        ratio = (n_files / total_files) * 100
        
        print(f"\n{split_name}:")
        print(f"  Total Files: {n_files:>6} ({ratio:>5.1f}%)")
        
        # Per-genus breakdown
        print(f"  Per Genus:")
        for genus in sorted(df["genus"].unique()):
            genus_count = len(df[df["genus"] == genus])
            print(f"    {genus:<12} {genus_count:>6} files")
        
        # Per-species breakdown
        print(f"  Per Species:")
        for species in sorted(df["species"].unique()):
            species_count = len(df[df["species"] == species])
            print(f"    {species:<25} {species_count:>6} files")
    
    print("\n" + "="*90)
    print(f"\nTotal Dataset: {total_files} files")
    print("="*90 + "\n")


def main():
    """Create train/val/test splits for Abuzz dataset."""
    
    print("\n" + "="*90)
    print(" "*30 + "ABUZZ DATASET SPLITTING")
    print("="*90)
    
    # Step 1: Collect all audio files
    print("\nScanning dataset...")
    files_data = collect_audio_files()
    
    print(f"Found {len(files_data)} audio files")
    
    # Per-genus file counts
    genus_counts = defaultdict(int)
    for f in files_data:
        genus_counts[f["genus"]] += 1
    
    print("\nFiles per genus:")
    for genus in sorted(genus_counts.keys()):
        print(f"  {genus:<12} {genus_counts[genus]:>6} files")
    
    # Step 2: Split data
    print("\nSplitting dataset (70% train, 15% val, 15% test)...")
    train_data, val_data, test_data = split_data(files_data)
    
    # Step 3: Create DataFrames
    print("Creating split files...")
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)
    
    # Step 4: Save CSVs
    output_dir = Path("data/splits")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / "abuzz_train.csv", index=False)
    val_df.to_csv(output_dir / "abuzz_val.csv", index=False)
    test_df.to_csv(output_dir / "abuzz_test.csv", index=False)
    
    print(f"\nSaved splits to {output_dir}/")
    print(f"  - abuzz_train.csv ({len(train_df)} files)")
    print(f"  - abuzz_val.csv ({len(val_df)} files)")
    print(f"  - abuzz_test.csv ({len(test_df)} files)")
    
    # Step 5: Print summary
    print_summary(train_df, val_df, test_df)
    
    print("Split creation complete!")


if __name__ == "__main__":
    main()
