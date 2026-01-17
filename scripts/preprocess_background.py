"""Preprocess Background dataset: extract diverse 1s windows using top-K energy + random-K sampling.

Usage:
    python scripts/preprocess_background.py
"""

from pathlib import Path
import warnings
import csv
import numpy as np
import soundfile as sf
from tqdm import tqdm
import random

warnings.filterwarnings('ignore')

AUDIO_EXTS = {'.wav', '.flac', '.mp3', '.m4a', '.aac', '.ogg'}

TARGET_SR = 16000
WINDOW_SEC = 1.0
TARGET_SAMPLES = int(TARGET_SR * WINDOW_SEC)

HOP_SEC = 1.0
RMS_THRESHOLD = 0.001
MAX_CANDIDATES_PER_FILE = 500  # Sample positions for long files
TOPK_ENERGY = 5
RANDOMK = 25
MAX_WINDOWS_PER_FILE = TOPK_ENERGY + RANDOMK
GLOBAL_MAX_WINDOWS = None
NORMALIZE_WINDOWS = False
CLIP_THRESHOLD = 0.999
RANDOM_SEED = 42


def compute_rms(audio):
    """Compute RMS energy of audio signal."""
    return np.sqrt(np.mean(audio ** 2))


def mosquito_score(window, sr=TARGET_SR):
    """Compute mosquito-band score: ratio of energy in 250-900 Hz band to total 0-4kHz."""
    FMAX = 4000
    BAND_LO, BAND_HI = 250, 900
    x = window.astype(np.float32)
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x))**2
    freqs = np.fft.rfftfreq(len(x), d=1.0/sr)
    
    mask_all = (freqs >= 0) & (freqs <= FMAX)
    mask_band = (freqs >= BAND_LO) & (freqs <= BAND_HI)
    
    denom = spec[mask_all].sum() + 1e-12
    numer = spec[mask_band].sum()
    return float(numer / denom)


def load_audio_smart(filepath, target_sr=TARGET_SR):
    """
    Load audio file with smart handling:
    - Fast path if already 16kHz mono WAV
    - Resample only if needed
    - Fallback to librosa for MP3/M4A if soundfile fails
    
    Returns:
        audio (np.ndarray): mono audio, float32, at target_sr
        sr (int): sample rate (should be target_sr)
    """
    try:
        # Try soundfile first (fast for WAV/FLAC)
        audio, sr = sf.read(filepath, dtype='float32')
        
        # Convert to mono if needed
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        
        # Resample if needed
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        return audio, sr
    
    except Exception as e:
        # Fallback for MP3/M4A
        try:
            import librosa
            audio, sr = librosa.load(filepath, sr=target_sr, mono=True)
            return audio, sr
        except Exception as e2:
            raise RuntimeError(f"Failed to load {filepath}: soundfile={e}, librosa={e2}")


def process_background_file(filepath, output_dir, input_base, stats, global_limit_remaining=None):
    """
    Process one background file: extract diverse windows using top-K + random-K.
    
    Args:
        filepath: Path to source audio file
        output_dir: Output directory
        input_base: Base input directory (for unique naming)
        stats: Dictionary to accumulate statistics
        global_limit_remaining: Remaining slots before hitting global limit (None = unlimited)
    
    Returns:
        windows_saved: Number of windows saved
    """
    try:
        # Load audio
        audio, sr = load_audio_smart(filepath, target_sr=TARGET_SR)
        
    except Exception as e:
        stats['errors'].append(f"{filepath.name}: {str(e)}")
        return 0
    
    # Calculate hop and window parameters
    hop_samples = int(HOP_SEC * TARGET_SR)
    
    # Sample candidate positions to avoid scanning every second of long files
    num_possible = (len(audio) - TARGET_SAMPLES) // hop_samples + 1
    if num_possible <= 0:
        return 0
    
    # Pick candidate indices
    if num_possible > MAX_CANDIDATES_PER_FILE:
        cand_idxs = sorted(np.random.choice(num_possible, MAX_CANDIDATES_PER_FILE, replace=False))
    else:
        cand_idxs = list(range(num_possible))
    
    candidates = []
    clipping_warnings = 0
    
    # Generate unique prefix from relative path
    try:
        rel_path = filepath.relative_to(input_base)
        file_prefix = str(rel_path.with_suffix('')).replace('\\', '_').replace('/', '_').replace(' ', '_')
    except ValueError:
        file_prefix = filepath.stem.replace(' ', '_')
    
    for j, widx in enumerate(cand_idxs):
        start_sample = widx * hop_samples
        window = audio[start_sample : start_sample + TARGET_SAMPLES]
        
        if len(window) < TARGET_SAMPLES:
            continue
        
        rms = compute_rms(window)
        if rms < RMS_THRESHOLD:
            continue
        
        # Reject mosquito-like background (fans, motors)
        if mosquito_score(window) > 0.12:
            continue
        
        peak = np.abs(window).max()
        if peak >= CLIP_THRESHOLD:
            clipping_warnings += 1
        
        candidates.append((window, rms, widx))
    
    if len(candidates) == 0:
        return 0
    
    stats['total_candidates'] += len(candidates)
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Filter impulse spikes (crest factor > 20)
    filtered = [c for c in candidates if (np.abs(c[0]).max() / (c[1] + 1e-8)) < 20]
    if len(filtered) >= TOPK_ENERGY:
        candidates = filtered
    
    max_to_save = MAX_WINDOWS_PER_FILE
    if global_limit_remaining is not None:
        max_to_save = min(max_to_save, global_limit_remaining)
    
    topk = min(TOPK_ENERGY, len(candidates), max_to_save)
    selected_topk = candidates[:topk]
    remaining_candidates = candidates[topk:]
    
    remaining_slots = max_to_save - topk
    if len(remaining_candidates) > 0 and remaining_slots > 0:
        randomk = min(RANDOMK, len(remaining_candidates), remaining_slots)
        selected_random = random.sample(remaining_candidates, randomk)
    else:
        selected_random = []
    
    selected = selected_topk + selected_random
    selected.sort(key=lambda x: x[2])
    
    windows_saved = 0
    for window, rms, widx in selected:
        output_filename = f"Background__{file_prefix}__idx{widx:06d}.wav"
        output_path = output_dir / output_filename
        sf.write(output_path, window, TARGET_SR, subtype='PCM_16')
        
        # Record in manifest
        rel_path = output_path.relative_to(Path('data/processed'))
        stats['manifest'].append({
            'path': str(rel_path).replace('\\', '/'),
            'label': 'Background',
            'source': 'Background',
            'species': 'Background',
            'origin': file_prefix,
            'rms': f"{rms:.6f}"
        })
        
        windows_saved += 1
    
    if clipping_warnings > 0:
        stats['clipping_files'].append(f"{filepath.name}: {clipping_warnings} windows near clipping")
    
    return windows_saved


def preprocess_background():
    """Main preprocessing for Background dataset."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    input_base = Path('data/raw/Background')
    output_base = Path('data/processed/Background/Background')
    
    if not input_base.exists():
        print(f"ERROR: Input directory not found: {input_base}")
        return
    
    print("="*80)
    print(" "*26 + "BACKGROUND PREPROCESSING")
    print("="*80)
    print(f"Input:  {input_base.absolute()}")
    print(f"Output: {output_base.absolute()}")
    print(f"\nConfiguration:")
    print(f"  Target sample rate:     {TARGET_SR} Hz")
    print(f"  Window duration:        {WINDOW_SEC} s ({TARGET_SAMPLES} samples)")
    print(f"  Hop duration:           {HOP_SEC} s (non-overlapping)")
    print(f"  RMS threshold:          {RMS_THRESHOLD} (permissive silence gate)")
    print(f"  Top-K by energy:        {TOPK_ENERGY} (captures events)")
    print(f"  Random-K from rest:     {RANDOMK} (captures ambience)")
    print(f"  Max windows per file:   {MAX_WINDOWS_PER_FILE}")
    print(f"  Global max windows:     {GLOBAL_MAX_WINDOWS if GLOBAL_MAX_WINDOWS else 'Unlimited (clamped per file)'}")
    print(f"  Normalize windows:      {NORMALIZE_WINDOWS}")
    print(f"  Clipping threshold:     {CLIP_THRESHOLD}")
    print(f"  Random seed:            {RANDOM_SEED}")
    print("="*80 + "\n")
    
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Clear existing output files to avoid duplicates
    for wav_file in output_base.glob('*.wav'):
        wav_file.unlink()
    
    stats = {
        'windows_created': 0,
        'files_processed': 0,
        'total_candidates': 0,
        'errors': [],
        'clipping_files': [],
        'manifest': []  # CSV records
    }
    
    audio_files = sorted([f for f in input_base.rglob('*') 
                         if f.is_file() and f.suffix.lower() in AUDIO_EXTS])
    
    if len(audio_files) == 0:
        print("ERROR: No audio files found in input path.")
        return
    
    print(f"Found {len(audio_files)} audio files")
    print(f"Processing with top-{TOPK_ENERGY} energy + random-{RANDOMK} strategy\n")
    
    total_windows = 0
    
    for audio_file in tqdm(audio_files, desc="  Processing", ncols=100):
        if GLOBAL_MAX_WINDOWS and total_windows >= GLOBAL_MAX_WINDOWS:
            print(f"\n  Reached global limit of {GLOBAL_MAX_WINDOWS} windows")
            break
        
        global_limit_remaining = None
        if GLOBAL_MAX_WINDOWS:
            global_limit_remaining = GLOBAL_MAX_WINDOWS - total_windows
        
        windows_saved = process_background_file(
            filepath=audio_file,
            output_dir=output_base,
            input_base=input_base,
            stats=stats,
            global_limit_remaining=global_limit_remaining
        )
        
        total_windows += windows_saved
        if windows_saved > 0:
            stats['files_processed'] += 1
    
    stats['windows_created'] = total_windows
    
    print("\n" + "="*80)
    print(" "*30 + "SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal:")
    print(f"  Windows saved:          {stats['windows_created']}")
    print(f"  Files processed:        {stats['files_processed']} (produced windows)")
    print(f"  Total files scanned:    {len(audio_files)}")
    
    if stats['files_processed'] > 0:
        avg_windows_per_file = stats['windows_created'] / stats['files_processed']
        print(f"  Avg windows per file:   {avg_windows_per_file:.1f}")
    
    if len(stats['clipping_files']) > 0:
        print(f"\nClipping warnings:      {len(stats['clipping_files'])} files had windows near clipping")
        if len(stats['clipping_files']) <= 10:
            for warning in stats['clipping_files']:
                print(f"  - {warning}")
        else:
            for warning in stats['clipping_files'][:5]:
                print(f"  - {warning}")
            print(f"  ... and {len(stats['clipping_files']) - 5} more files")
    
    if len(stats['errors']) > 0:
        print(f"\nErrors encountered:     {len(stats['errors'])}")
        if len(stats['errors']) <= 10:
            for error in stats['errors']:
                print(f"  - {error}")
        else:
            for error in stats['errors'][:5]:
                print(f"  - {error}")
            print(f"  ... and {len(stats['errors']) - 5} more errors")
    
    print("="*80)
    print("\nPreprocessing complete!")
    print(f"Output saved to: {output_base.absolute()}")
    
    # Write manifest CSV
    manifest_path = Path('data/splits/background_manifest.csv')
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if stats['manifest']:
        with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['path', 'label', 'source', 'species', 'origin', 'rms']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(stats['manifest'])
        print(f"Manifest saved to: {manifest_path.absolute()}")
    
    print()


def main():
    preprocess_background()


if __name__ == "__main__":
    main()

