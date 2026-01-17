"""Preprocess ABUZZ dataset: extract 1s windows scored by mosquito-band energy.

Usage:
    python scripts/preprocess_abuzz.py
"""

from pathlib import Path
from collections import defaultdict
import warnings
import csv
import numpy as np
import soundfile as sf
from tqdm import tqdm

warnings.filterwarnings('ignore')

AUDIO_EXTS = {'.wav', '.flac', '.mp3', '.m4a', '.aac', '.ogg'}

TARGET_SR = 16000
WINDOW_SEC = 1.0
TARGET_SAMPLES = int(TARGET_SR * WINDOW_SEC)

CLASS_MAPPING = {
    'Aedes aegypti': 'Aedes',
    'Aedes albopictus': 'Aedes',
    'Anopheles arabiensis': 'Anopheles',
    'Anopheles gambiae': 'Anopheles',
    'Culex pipiens': 'Culex',
    'Culex quinquefasciatus': 'Culex',
}

FMAX = 4000
BAND_LO, BAND_HI = 250, 1200  # Mosquito wingbeat frequency range (wider for harmonics)
BAND_ENERGY_MIN = 1e-6  # Minimum absolute band energy (prevents high-ratio but quiet windows)
HOP_SEC = 0.5
RMS_THRESHOLD = 0.001
MOSQUITO_SCORE_MIN = 0.02  # Minimum ratio threshold
K_WINDOWS_PER_FILE = 5
NORMALIZE_WINDOWS = False
CLIP_THRESHOLD = 0.999


def compute_rms(audio):
    """Compute RMS energy of audio signal."""
    return np.sqrt(np.mean(audio ** 2))


def mosquito_score(window, sr=TARGET_SR):
    """Compute mosquito-band score: ratio and absolute energy in mosquito band.
    
    Returns:
        tuple: (ratio, band_energy) where ratio is band/total and band_energy is absolute
    """
    x = window.astype(np.float32)
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x))**2
    freqs = np.fft.rfftfreq(len(x), d=1.0/sr)
    
    mask_all = (freqs >= 0) & (freqs <= FMAX)
    mask_band = (freqs >= BAND_LO) & (freqs <= BAND_HI)
    
    band_energy = spec[mask_band].sum()
    all_energy = spec[mask_all].sum()
    ratio = float(band_energy / (all_energy + 1e-12))
    
    return ratio, band_energy


def normalize_window(audio, headroom=0.95):
    """Peak normalize audio window to prevent clipping."""
    peak = np.abs(audio).max()
    if peak > 1e-6:  # Avoid division by zero
        return audio * (headroom / peak)
    return audio


def load_audio_smart(filepath, target_sr=TARGET_SR):
    """
    Load audio file with smart handling:
    - Fast path if already 16kHz mono WAV
    - Resample only if needed
    
    Returns:
        audio (np.ndarray): mono audio, float32, at target_sr
        sr (int): sample rate (should be target_sr)
    """
    try:
        # Load with soundfile
        audio, sr = sf.read(filepath, dtype='float32')
        
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        return audio, sr
    except Exception as e:
        raise RuntimeError(f"Failed to load {filepath}: {e}")


def process_abuzz_file(filepath, class_name, species_name, output_dir, input_base, stats):
    """
    Process one ABUZZ file: extract windows, score them, and select top K.
    
    Args:
        filepath: Path to source audio file
        class_name: Output class name (Aedes/Anopheles/Culex)
        species_name: Original species name
        output_dir: Output directory for this class
        input_base: Base input directory (for collision-safe origin)
        stats: Dictionary to accumulate statistics
    
    Returns:
        Tuple of (windows_saved, total_candidates, windows_rejected)
    """
    try:
        # Load audio
        audio, sr = load_audio_smart(filepath, target_sr=TARGET_SR)
        
    except Exception as e:
        stats['errors'].append(f"{filepath.name}: {str(e)}")
        return 0, 0, 0
    
    hop_samples = int(HOP_SEC * TARGET_SR)
    candidates = []
    clipping_warnings = 0
    file_stem = filepath.stem.replace(' ', '_')
    
    # Track filtering stats
    total_windows = 0
    passed_rms = 0
    passed_score = 0
    
    for idx, start_sample in enumerate(range(0, len(audio) - TARGET_SAMPLES + 1, hop_samples)):
        window = audio[start_sample : start_sample + TARGET_SAMPLES]
        
        if len(window) < TARGET_SAMPLES:
            continue
        
        total_windows += 1
        
        rms = compute_rms(window)
        if rms < RMS_THRESHOLD:
            continue
        
        passed_rms += 1
        
        ratio, band_energy = mosquito_score(window, sr=TARGET_SR)
        
        # CRITICAL: Require BOTH high ratio AND sufficient absolute energy
        # This blocks "quiet but high-ratio" windows (e.g., faint hums)
        if ratio < MOSQUITO_SCORE_MIN or band_energy < BAND_ENERGY_MIN:
            continue
        
        passed_score += 1
        
        peak = np.abs(window).max()
        if peak >= CLIP_THRESHOLD:
            clipping_warnings += 1
        
        candidates.append((window, ratio, band_energy, rms, idx))
    
    total_candidates = len(candidates)
    
    # Track score distribution
    if len(candidates) > 0:
        scores = [c[1] for c in candidates]
        stats['score_distribution'].extend(scores)
    
    if len(candidates) > K_WINDOWS_PER_FILE:
        candidates.sort(key=lambda x: x[1], reverse=True)  # Sort by ratio score
        selected = candidates[:K_WINDOWS_PER_FILE]
        
        # Add loudest window as an extra diversity window (often noisy)
        loudest = max(candidates, key=lambda x: x[3])  # By RMS
        if loudest not in selected:
            selected.append(loudest)
        
        # Add median-score window for harder positives (prevents overfitting to cleanest)
        if len(candidates) >= 3:
            by_score = sorted(candidates, key=lambda x: x[1])  # Sort by ratio ascending
            median = by_score[len(by_score) // 2]
            if median not in selected:
                selected.append(median)
        
        # Note: Can save up to K+2 windows (K top + loudest + median if not already included)
        # To enforce hard cap: selected = selected[:K_WINDOWS_PER_FILE]
        
        windows_rejected = len(candidates) - len(selected)
    else:
        selected = candidates
        windows_rejected = 0
    
    selected.sort(key=lambda x: x[4])  # Sort by index for temporal order
    
    # Create collision-safe origin key from relative path
    origin_key = str(filepath.relative_to(input_base)).replace('\\', '/')
    
    windows_saved = 0
    for window, ratio, band_energy, rms, idx in selected:
        if NORMALIZE_WINDOWS:
            window = normalize_window(window)
        
        output_filename = f"{species_name.replace(' ', '_')}__{file_stem}__idx{idx:06d}.wav"
        output_path = output_dir / output_filename
        sf.write(output_path, window, TARGET_SR, subtype='PCM_16')
        
        # Record in manifest
        rel_path = output_path.relative_to(Path('data/processed'))
        stats['manifest'].append({
            'path': str(rel_path).replace('\\', '/'),
            'label': class_name,
            'source': 'Abuzz',
            'species': species_name,
            'origin': origin_key,
            'score': f"{ratio:.4f}",
            'rms': f"{rms:.6f}"
        })
        
        windows_saved += 1
    
    if clipping_warnings > 0:
        stats['clipping_files'].append(f"{filepath.name}: {clipping_warnings} windows near clipping")
    
    # Store filtering stats
    stats['file_stats'].append({
        'file': filepath.name,
        'total_windows': total_windows,
        'passed_rms': passed_rms,
        'passed_score': passed_score,
        'candidates': total_candidates,
        'saved': windows_saved
    })
    
    return windows_saved, total_candidates, windows_rejected


def preprocess_abuzz():
    """Main preprocessing for ABUZZ dataset."""
    input_base = Path('data/raw/Abuzz')
    output_base = Path('data/processed/Abuzz')
    
    if not input_base.exists():
        print(f"ERROR: Input directory not found: {input_base}")
        return
    
    # Clear existing output files to avoid duplicates
    if output_base.exists():
        import shutil
        shutil.rmtree(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(" "*28 + "ABUZZ PREPROCESSING")
    print("="*80)
    print(f"Input:  {input_base.absolute()}")
    print(f"Output: {output_base.absolute()}")
    print(f"\nConfiguration:")
    print(f"  Target sample rate:     {TARGET_SR} Hz")
    print(f"  Window duration:        {WINDOW_SEC} s ({TARGET_SAMPLES} samples)")
    print(f"  Hop duration:           {HOP_SEC} s (50% overlap)")
    print(f"  RMS threshold:          {RMS_THRESHOLD}")
    print(f"  Mosquito band:          {BAND_LO}-{BAND_HI} Hz (out of 0-{FMAX} Hz)")
    print(f"  Score ratio threshold:  {MOSQUITO_SCORE_MIN}")
    print(f"  Band energy threshold:  {BAND_ENERGY_MIN}")
    print(f"  Top-K per file:         {K_WINDOWS_PER_FILE} (+ up to 2 diversity: loudest + median)")
    print(f"  Normalize windows:      {NORMALIZE_WINDOWS}")
    print(f"  Clipping threshold:     {CLIP_THRESHOLD}")
    print("="*80 + "\n")
    
    stats = {
        'windows_created': defaultdict(int),
        'files_processed': defaultdict(int),
        'total_candidates': 0,
        'windows_rejected': 0,
        'errors': [],
        'clipping_files': [],
        'manifest': [],  # CSV records - CRITICAL: Split by 'origin' (file stem), not by window!
        'score_distribution': [],
        'file_stats': []
    }
    
    species_dirs = sorted([d for d in input_base.iterdir() if d.is_dir()])
    
    if len(species_dirs) == 0:
        print("ERROR: No species directories found in input path.")
        return
    
    for species_dir in species_dirs:
        species_name = species_dir.name
        
        if species_name not in CLASS_MAPPING:
            print(f"WARNING: Unknown species '{species_name}' - skipping")
            continue
        
        class_name = CLASS_MAPPING[species_name]
        
        output_dir = output_base / class_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        audio_files = sorted([f for f in species_dir.iterdir() 
                             if f.is_file() and f.suffix.lower() in AUDIO_EXTS])
        
        if len(audio_files) == 0:
            print(f"WARNING: No audio files found for {species_name}")
            continue
        
        print(f"Processing {species_name} -> {class_name}")
        print(f"  Found {len(audio_files)} audio files")
        
        species_windows = 0
        species_candidates = 0
        species_rejected = 0
        
        for audio_file in tqdm(audio_files, desc=f"  {species_name}", ncols=100):
            windows, candidates, rejected = process_abuzz_file(
                filepath=audio_file,
                class_name=class_name,
                species_name=species_name,
                output_dir=output_dir,
                input_base=input_base,
                stats=stats
            )
            species_windows += windows
            species_candidates += candidates
            species_rejected += rejected
        
        stats['windows_created'][class_name] += species_windows
        stats['files_processed'][class_name] += len(audio_files)
        stats['total_candidates'] += species_candidates
        stats['windows_rejected'] += species_rejected
        
        avg_windows = species_windows / len(audio_files) if len(audio_files) > 0 else 0
        avg_candidates = species_candidates / len(audio_files) if len(audio_files) > 0 else 0
        print(f"  Saved {species_windows} windows from {len(audio_files)} files " +
              f"(avg: {avg_windows:.1f} saved/file, {avg_candidates:.1f} candidates/file)\n")
    
    print("\n" + "="*80)
    print(" "*30 + "SUMMARY STATISTICS")
    print("="*80)
    
    total_windows = sum(stats['windows_created'].values())
    total_files = sum(stats['files_processed'].values())
    
    print("\nWindows created by class:")
    for class_name in sorted(stats['windows_created'].keys()):
        count = stats['windows_created'][class_name]
        pct = (count / total_windows * 100) if total_windows > 0 else 0
        print(f"  {class_name:15s} {count:6d} windows ({pct:5.1f}%)")
    
    print(f"\nTotal:")
    print(f"  Windows saved:          {total_windows}")
    print(f"  Files processed:        {total_files}")
    print(f"  Total candidates:       {stats['total_candidates']}")
    print(f"  Windows rejected:       {stats['windows_rejected']} (kept top {K_WINDOWS_PER_FILE} per file)")
    
    if total_files > 0:
        avg_windows_per_file = total_windows / total_files
        avg_candidates_per_file = stats['total_candidates'] / total_files
        print(f"  Avg saved per file:     {avg_windows_per_file:.1f}")
        print(f"  Avg candidates per file: {avg_candidates_per_file:.1f}")
    
    if len(stats['clipping_files']) > 0:
        print(f"\nClipping warnings:      {len(stats['clipping_files'])} files had windows near clipping")
        if len(stats['clipping_files']) <= 10:
            for warning in stats['clipping_files']:
                print(f"  - {warning}")
        else:
            for warning in stats['clipping_files'][:5]:
                print(f"  - {warning}")
            print(f"  ... and {len(stats['clipping_files']) - 5} more files")
        print("  (This is common with ABUZZ mobile recordings; usually not a problem)")
    
    if len(stats['errors']) > 0:
        print(f"\nErrors encountered:     {len(stats['errors'])}")
        if len(stats['errors']) <= 10:
            for error in stats['errors']:
                print(f"  - {error}")
        else:
            for error in stats['errors'][:5]:
                print(f"  - {error}")
            print(f"  ... and {len(stats['errors']) - 5} more errors")
    
    # CRITICAL: Filtering acceptance analysis
    print("\n" + "="*80)
    print(" "*25 + "FILTERING ACCEPTANCE ANALYSIS")
    print("="*80)
    
    if stats['file_stats']:
        total_windows_scanned = sum(f['total_windows'] for f in stats['file_stats'])
        total_passed_rms = sum(f['passed_rms'] for f in stats['file_stats'])
        total_passed_score = sum(f['passed_score'] for f in stats['file_stats'])
        
        rms_rate = (total_passed_rms / total_windows_scanned * 100) if total_windows_scanned > 0 else 0
        score_rate = (total_passed_score / total_passed_rms * 100) if total_passed_rms > 0 else 0
        overall_rate = (total_passed_score / total_windows_scanned * 100) if total_windows_scanned > 0 else 0
        
        print(f"\nWindows scanned:        {total_windows_scanned}")
        print(f"Passed RMS filter:      {total_passed_rms} ({rms_rate:.1f}%)")
        print(f"Passed score filter:    {total_passed_score} ({score_rate:.1f}% of RMS-passed)")
        print(f"Overall acceptance:     {overall_rate:.1f}%")
        
        # Files with no candidates
        zero_candidate_files = [f['file'] for f in stats['file_stats'] if f['candidates'] == 0]
        if zero_candidate_files:
            print(f"\nFiles with 0 candidates: {len(zero_candidate_files)}/{len(stats['file_stats'])}")
            if len(zero_candidate_files) <= 5:
                for fname in zero_candidate_files:
                    print(f"  - {fname}")
            else:
                for fname in zero_candidate_files[:3]:
                    print(f"  - {fname}")
                print(f"  ... and {len(zero_candidate_files) - 3} more")
            print("  → If many files have 0 candidates, thresholds may be too strict!")
    
    # Score distribution
    if stats['score_distribution']:
        scores = np.array(stats['score_distribution'])
        print(f"\nScore distribution:")
        print(f"  Min:     {scores.min():.4f}")
        print(f"  Median:  {np.median(scores):.4f}")
        print(f"  Mean:    {scores.mean():.4f}")
        print(f"  Max:     {scores.max():.4f}")
        print(f"  Std:     {scores.std():.4f}")
    
    print("="*80)
    print("\nPreprocessing complete!")
    print(f"Output saved to: {output_base.absolute()}")
    
    # Write manifest CSV
    manifest_path = Path('data/splits/abuzz_manifest.csv')
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if stats['manifest']:
        with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['path', 'label', 'source', 'species', 'origin', 'score', 'rms']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(stats['manifest'])
        print(f"Manifest saved to: {manifest_path.absolute()}")
    
    print()


def main():
    preprocess_abuzz()


if __name__ == "__main__":
    main()
