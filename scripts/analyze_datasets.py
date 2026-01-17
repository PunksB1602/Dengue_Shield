from pathlib import Path
from collections import defaultdict
import pandas as pd

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"}


def count_recordings_recursive(species_dir, max_depth=2):
    """
    Find recording folders by D_YY_MM_DD_HH_MM_SS pattern.
    Returns list with path, nesting info.
    """
    recordings = []
    
    def find_recordings(path, current_depth=0, parent_folder=None):
        if current_depth > max_depth:
            return
        
        for item in path.iterdir():
            if item.is_dir():
                # Recording folders start with D_ and have at least date components
                if item.name.startswith('D_') and len(item.name.split('_')) >= 7:
                    recordings.append({
                        'path': item,
                        'nested': current_depth > 0,
                        'parent': parent_folder
                    })
                else:
                    # Keep searching deeper
                    find_recordings(item, current_depth + 1, item.name)
    
    find_recordings(species_dir)
    return recordings


def get_recording_dirs(species_dir, max_depth=4):
    """Try D_* pattern first, fallback to any folder with audio files."""
    # Try pattern-based detection
    rec_info = count_recordings_recursive(species_dir, max_depth=max_depth)
    rec_dirs = [r["path"] for r in rec_info]

    # Fallback: just grab any parent folder containing audio
    if len(rec_dirs) == 0:
        rec_dirs = sorted({f.parent for f in species_dir.rglob("*")
                          if f.is_file() and f.suffix.lower() in AUDIO_EXTS})
    return rec_dirs


def analyze_abuzz(base_path):
    """Analyze Abuzz dataset (flat structure: species/*.wav)."""
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Abuzz path not found: {base_path}")
        return None
    
    species_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
    
    print("\n" + "="*90)
    print(" "*35 + "ABUZZ DATASET")
    print("="*90)
    print(f"Path: {base_path.absolute()}\n")
    
    results = []
    total_files = 0
    total_size = 0
    
    for species_dir in species_dirs:
        species_name = species_dir.name
        
        files = [f for f in species_dir.iterdir()
                 if f.is_file() and f.suffix.lower() in AUDIO_EXTS]
        
        file_count = len(files)
        file_types = defaultdict(int)
        size_bytes = 0
        
        for f in files:
            ext = f.suffix.lower() if f.suffix else '(no ext)'
            file_types[ext] += 1
            try:
                size_bytes += f.stat().st_size
            except OSError:
                pass
        
        size_mb = size_bytes / (1024 * 1024)
        
        results.append({
            'Species': species_name,
            'Files': file_count,
            'Size (MB)': round(size_mb, 2),
            'File Types': ', '.join(sorted(file_types.keys())),
            '% of Total': 0  # calculated below
        })
        
        total_files += file_count
        total_size += size_mb
    
    # Calculate percentages
    for result in results:
        if total_files > 0:
            result['% of Total'] = round((result['Files'] / total_files) * 100, 2)
    
    # Create DataFrame and print table
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    print("\n" + "="*90)
    print(" "*35 + "SUMMARY STATISTICS")
    print("="*90)
    print(f"Total Species:              {len(species_dirs)}")
    print(f"Total Files:                {total_files}")
    print(f"Total Dataset Size:         {total_size:.2f} MB ({total_size/1024:.2f} GB)")
    
    if len(species_dirs) > 0:
        print(f"\nAverage per Species:")
        print(f"  - Files:                  {total_files/len(species_dirs):.1f}")
        print(f"  - Size:                   {(total_size/len(species_dirs)):.2f} MB")
    
    print("="*90)
    
    return df


def analyze_wingsbeats(base_path):
    """Analyze Wingsbeats dataset (nested structure with recording folders)."""
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Wingsbeats path not found: {base_path}")
        return None
    
    species_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
    
    print("\n" + "="*90)
    print(" "*33 + "WINGSBEATS DATASET")
    print("="*90)
    print(f"Path: {base_path.absolute()}\n")
    
    results = []
    total_recordings = 0
    total_files = 0
    total_size = 0
    
    for species_dir in species_dirs:
        species_name = species_dir.name
        
        # Get recording folders (with pattern detection + fallback)
        recordings = get_recording_dirs(species_dir)
        
        # Count audio files
        file_count = 0
        file_types = defaultdict(int)
        size_bytes = 0
        
        for recording in recordings:
            files = [f for f in recording.rglob("*")
                     if f.is_file() and f.suffix.lower() in AUDIO_EXTS]
            
            file_count += len(files)
            
            for f in files:
                ext = f.suffix.lower() if f.suffix else '(no ext)'
                file_types[ext] += 1
                try:
                    size_bytes += f.stat().st_size
                except OSError:
                    pass
        
        size_mb = size_bytes / (1024 * 1024)
        rec_count = len(recordings)
        
        results.append({
            'Species': species_name,
            'Recordings': rec_count,
            'Files': file_count,
            'Size (MB)': round(size_mb, 2),
            'File Types': ', '.join(sorted(file_types.keys())),
            '% of Total': 0  # calculated below
        })
        
        total_recordings += rec_count
        total_files += file_count
        total_size += size_mb
    
    # Calculate percentages
    for result in results:
        if total_files > 0:
            result['% of Total'] = round((result['Files'] / total_files) * 100, 2)
    
    # Create DataFrame and print table
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    print("\n" + "="*90)
    print(" "*35 + "SUMMARY STATISTICS")
    print("="*90)
    print(f"Total Species:              {len(species_dirs)}")
    print(f"Total Recordings:           {total_recordings}")
    print(f"Total Files:                {total_files}")
    print(f"Total Dataset Size:         {total_size:.2f} MB ({total_size/1024:.2f} GB)")
    
    if len(species_dirs) > 0:
        print(f"\nAverage per Species:")
        print(f"  - Recordings:             {total_recordings/len(species_dirs):.1f}")
        print(f"  - Files:                  {total_files/len(species_dirs):.1f}")
        print(f"  - Size:                   {(total_size/len(species_dirs)):.2f} MB")
    
    print("="*90)
    
    return df


def analyze_background(base_path):
    """Analyze Background dataset (audio files for background noise)."""
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Background path not found: {base_path}")
        return None
    
    print("\n" + "="*90)
    print(" "*33 + "BACKGROUND DATASET")
    print("="*90)
    print(f"Path: {base_path.absolute()}\n")
    
    # Check if there are subdirectories or just flat files
    subdirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
    
    results = []
    total_files = 0
    total_size = 0
    
    if len(subdirs) > 0:
        # Has subdirectories (e.g., different background types)
        for subdir in subdirs:
            subdir_name = subdir.name
            
            files = [f for f in subdir.rglob("*")
                     if f.is_file() and f.suffix.lower() in AUDIO_EXTS]
            
            file_count = len(files)
            file_types = defaultdict(int)
            size_bytes = 0
            
            for f in files:
                ext = f.suffix.lower() if f.suffix else '(no ext)'
                file_types[ext] += 1
                try:
                    size_bytes += f.stat().st_size
                except OSError:
                    pass
            
            size_mb = size_bytes / (1024 * 1024)
            
            results.append({
                'Category': subdir_name,
                'Files': file_count,
                'Size (MB)': round(size_mb, 2),
                'File Types': ', '.join(sorted(file_types.keys())),
                '% of Total': 0  # calculated below
            })
            
            total_files += file_count
            total_size += size_mb
    else:
        # Flat structure - all files directly in Background folder
        files = [f for f in base_path.rglob("*")
                 if f.is_file() and f.suffix.lower() in AUDIO_EXTS]
        
        file_count = len(files)
        file_types = defaultdict(int)
        size_bytes = 0
        
        for f in files:
            ext = f.suffix.lower() if f.suffix else '(no ext)'
            file_types[ext] += 1
            try:
                size_bytes += f.stat().st_size
            except OSError:
                pass
        
        size_mb = size_bytes / (1024 * 1024)
        
        results.append({
            'Category': 'Background',
            'Files': file_count,
            'Size (MB)': round(size_mb, 2),
            'File Types': ', '.join(sorted(file_types.keys())),
            '% of Total': 100.0
        })
        
        total_files = file_count
        total_size = size_mb
    
    # Calculate percentages
    for result in results:
        if total_files > 0:
            result['% of Total'] = round((result['Files'] / total_files) * 100, 2)
    
    # Create DataFrame and print table
    if len(results) > 0:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
        
        print("\n" + "="*90)
        print(" "*35 + "SUMMARY STATISTICS")
        print("="*90)
        print(f"Total Categories:           {len(results)}")
        print(f"Total Files:                {total_files}")
        print(f"Total Dataset Size:         {total_size:.2f} MB ({total_size/1024:.2f} GB)")
        
        if len(results) > 1:
            print(f"\nAverage per Category:")
            print(f"  - Files:                  {total_files/len(results):.1f}")
            print(f"  - Size:                   {(total_size/len(results)):.2f} MB")
        
        print("="*90)
        
        return df
    else:
        print("No audio files found in Background dataset.")
        print("="*90)
        return None


def main():
    """Analyze both datasets and export summary."""
    
    Path("data/splits").mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*90)
    print(" "*25 + "DENGUE SHIELD - DATASET ANALYSIS")
    print("="*90)
    
    # Analyze all datasets
    abuzz_df = analyze_abuzz("data/raw/Abuzz")
    wingsbeats_df = analyze_wingsbeats("data/raw/Wingsbeats")
    background_df = analyze_background("data/raw/Background")
    
    # Combined summary
    print("\n" + "="*90)
    print(" "*35 + "COMBINED SUMMARY")
    print("="*90)
    
    total_files = 0
    total_size = 0
    species_sets = set()
    
    if abuzz_df is not None and len(abuzz_df) > 0:
        total_files += abuzz_df['Files'].sum()
        total_size += abuzz_df['Size (MB)'].sum()
        species_sets.update(abuzz_df['Species'].tolist())
        print(f"  Abuzz:       {len(abuzz_df)} species, {abuzz_df['Files'].sum()} files")
    
    if wingsbeats_df is not None and len(wingsbeats_df) > 0:
        total_files += wingsbeats_df['Files'].sum()
        total_size += wingsbeats_df['Size (MB)'].sum()
        species_sets.update(wingsbeats_df['Species'].tolist())
        print(f"  Wingsbeats:   {len(wingsbeats_df)} species, {wingsbeats_df['Files'].sum()} files")
    
    if background_df is not None and len(background_df) > 0:
        total_files += background_df['Files'].sum()
        total_size += background_df['Size (MB)'].sum()
        print(f"  Background:   {len(background_df)} categories, {background_df['Files'].sum()} files")
    
    print(f"\n  Total unique species: {len(species_sets)}")
    print(f"  Total files:          {total_files}")
    print(f"  Total size:           {total_size:.2f} MB ({total_size/1024:.2f} GB)")
    
    # Export CSVs with TOTAL and AVERAGE rows appended
    if abuzz_df is not None and len(abuzz_df) > 0:
        abuzz_export = abuzz_df.copy()
        summary_row = {
            'Species': 'TOTAL',
            'Files': abuzz_export['Files'].sum(),
            'Size (MB)': round(abuzz_export['Size (MB)'].sum(), 2),
            'File Types': 'N/A',
            '% of Total': 100.0
        }
        avg_row = {
            'Species': 'AVERAGE',
            'Files': round(abuzz_export['Files'].mean(), 1),
            'Size (MB)': round(abuzz_export['Size (MB)'].mean(), 2),
            'File Types': 'N/A',
            '% of Total': round(100.0 / len(abuzz_export), 2)
        }
        abuzz_export = pd.concat([abuzz_export, pd.DataFrame([summary_row, avg_row])], ignore_index=True)
        abuzz_export.to_csv("data/splits/abuzz_analysis.csv", index=False)
        print(f"\n  Exported: data/splits/abuzz_analysis.csv")
    
    if wingsbeats_df is not None and len(wingsbeats_df) > 0:
        wingsbeats_export = wingsbeats_df.copy()
        summary_row = {
            'Species': 'TOTAL',
            'Recordings': wingsbeats_export['Recordings'].sum(),
            'Files': wingsbeats_export['Files'].sum(),
            'Size (MB)': round(wingsbeats_export['Size (MB)'].sum(), 2),
            'File Types': 'N/A',
            '% of Total': 100.0
        }
        avg_row = {
            'Species': 'AVERAGE',
            'Recordings': round(wingsbeats_export['Recordings'].mean(), 1),
            'Files': round(wingsbeats_export['Files'].mean(), 1),
            'Size (MB)': round(wingsbeats_export['Size (MB)'].mean(), 2),
            'File Types': 'N/A',
            '% of Total': round(100.0 / len(wingsbeats_export), 2)
        }
        wingsbeats_export = pd.concat([wingsbeats_export, pd.DataFrame([summary_row, avg_row])], ignore_index=True)
        wingsbeats_export.to_csv("data/splits/wingsbeats_analysis.csv", index=False)
        print(f"  Exported: data/splits/wingsbeats_analysis.csv")
    
    if background_df is not None and len(background_df) > 0:
        background_export = background_df.copy()
        summary_row = {
            'Category': 'TOTAL',
            'Files': background_export['Files'].sum(),
            'Size (MB)': round(background_export['Size (MB)'].sum(), 2),
            'File Types': 'N/A',
            '% of Total': 100.0
        }
        avg_row = {
            'Category': 'AVERAGE',
            'Files': round(background_export['Files'].mean(), 1),
            'Size (MB)': round(background_export['Size (MB)'].mean(), 2),
            'File Types': 'N/A',
            '% of Total': round(100.0 / len(background_export), 2)
        }
        background_export = pd.concat([background_export, pd.DataFrame([summary_row, avg_row])], ignore_index=True)
        background_export.to_csv("data/splits/background_analysis.csv", index=False)
        print(f"  Exported: data/splits/background_analysis.csv")
    
    print("="*90 + "\n")


if __name__ == "__main__":
    main()



