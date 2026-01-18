"""
Preprocess Background dataset: extract diverse 1s windows using top-K energy + random-K sampling.
Usage:
    python scripts/preprocess_background.py
"""

from pathlib import Path
import warnings
import csv
import hashlib
import numpy as np
import soundfile as sf
from tqdm import tqdm
import random

try:
    import librosa
except ImportError:
    librosa = None

warnings.filterwarnings('ignore')

AUDIO_EXTS = {'.wav', '.flac', '.mp3', '.m4a', '.aac', '.ogg'}

TARGET_SR = 16000
WINDOW_SEC = 1.0
TARGET_SAMPLES = int(TARGET_SR * WINDOW_SEC)

VERBOSE = False                # Enable verbose per-file logging
HOP_SEC = 0.5                  # Default hop for long files (adaptive per-file)
RMS_THRESHOLD = 0.0003         # Lower threshold for quiet ambience
MAX_CANDIDATES_PER_FILE = 4000 # Cap candidate windows for long files
TOPK_ENERGY = 5                # Number of highest-energy windows to select
RANDOMK = 115                  # Number of additional random windows
MAX_WINDOWS_PER_FILE = TOPK_ENERGY + RANDOMK
HARD_CAP_PER_FILE = 120        # Maximum windows per file regardless of MAX_WINDOWS_PER_FILE
GLOBAL_MAX_WINDOWS = None
NORMALIZE_WINDOWS = True
REJECT_MOSQUITO_LIKE = False
CLIP_THRESHOLD = 0.999
RANDOM_SEED = 42


def compute_rms(audio):
    """Root mean square energy."""
    return np.sqrt(np.mean(audio ** 2))


def mosquito_score(window, sr=TARGET_SR):
    """Mosquito-band energy ratio for filtering."""
    FMAX = 4000
    BAND_LO, BAND_HI = 250, 1200
    x = window.astype(np.float32)
    x = x - np.mean(x)
    w = np.hanning(len(x)).astype(np.float32)
    spec = np.abs(np.fft.rfft(x * w))**2
    freqs = np.fft.rfftfreq(len(x), d=1.0/sr)
    mask_all = (freqs >= 0) & (freqs <= FMAX)
    mask_band = (freqs >= BAND_LO) & (freqs <= BAND_HI)
    band_energy = spec[mask_band].sum()
    all_energy = spec[mask_all].sum()
    ratio = float(band_energy / (all_energy + 1e-12))
    return ratio, band_energy


def load_audio_smart(filepath, target_sr=TARGET_SR):
    """Robust audio loader: mono conversion and resampling if needed."""
    try:
        audio, sr = sf.read(filepath, dtype='float32', always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            if librosa is None:
                raise RuntimeError("librosa required for resampling but not installed")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return audio.astype(np.float32, copy=False), sr
    except Exception as e:
        try:
            if librosa is None:
                raise RuntimeError(f"librosa required but not installed: {filepath}")
            audio, sr = librosa.load(str(filepath), sr=target_sr, mono=True)
            return audio.astype(np.float32, copy=False), sr
        except Exception as e2:
            raise RuntimeError(f"Failed to load {filepath}: soundfile={e}, librosa={e2}")


def process_background_file(filepath, output_dir, input_base, stats, global_limit_remaining=None):
    """Extract windows from one file using top-K and random-K selection."""
    try:
        audio, sr = load_audio_smart(filepath, target_sr=TARGET_SR)
    except Exception as e:
        stats['errors'].append(f"{filepath.name}: {str(e)}")
        return 0
    # Per-file RNG for reproducibility
    try:
        rel_file_path = filepath.relative_to(input_base)
        origin_key = str(rel_file_path).replace('\\', '/')
    except ValueError:
        origin_key = filepath.name
    file_seed = int(hashlib.md5(origin_key.encode('utf-8')).hexdigest()[:8], 16)
    rng_np = np.random.RandomState(file_seed)
    rng_py = random.Random(file_seed)
    # Adaptive hop size based on file duration for better coverage
    duration_sec = len(audio) / sr
    # Initial bucketing by filename keywords
    name = filepath.name.lower()
    bucket_by_name = "hard" if any(k in name for k in ["voice", "fan", "ac", "motor", "hum", "compressor"]) else "easy"
    bucket = bucket_by_name  # Will be updated based on mosquito_ratio later
    
    # Track per-file bucket assignment
    if bucket_by_name == "hard":
        stats['hard_files_by_name'] += 1
    ratio_hard_file = False  # Flag for ratio-based hard detection
    if duration_sec <= 10:
        hop_sec = 0.10
    elif duration_sec <= 30:
        hop_sec = 0.25
    else:
        hop_sec = HOP_SEC
    if VERBOSE:
        print(f"  {filepath.name}: duration={duration_sec:.1f}s, hop={hop_sec}s")
    hop_samples = max(1, int(hop_sec * sr))
    num_possible = (len(audio) - TARGET_SAMPLES) // hop_samples + 1
    if num_possible <= 0:
        return 0
    # Block-based sampling for reproducibility on long files
    if num_possible > MAX_CANDIDATES_PER_FILE:
        import math
        num_blocks = 20
        block_size = math.ceil(num_possible / num_blocks)
        samples_per_block = max(1, MAX_CANDIDATES_PER_FILE // num_blocks)
        cand_idxs = []
        for block_idx in range(num_blocks):
            block_start = block_idx * block_size
            block_end = min((block_idx + 1) * block_size, num_possible)
            if block_end > block_start:
                block_samples = min(samples_per_block, block_end - block_start)
                block_choices = rng_np.choice(
                    range(block_start, block_end), 
                    block_samples, 
                    replace=False
                )
                cand_idxs.extend(block_choices)
        cand_idxs = sorted(cand_idxs)
    else:
        cand_idxs = list(range(num_possible))
    candidates = []
    clipping_warnings = 0
    rejected_rms = 0
    rejected_peak = 0
    rejected_easy_high_mosquito = 0
    # Use origin_key hash to prevent collisions and long paths
    fid = hashlib.md5(origin_key.encode('utf-8')).hexdigest()[:8]
    file_prefix = origin_key.replace('/', '_').replace(' ', '_')[-120:]
    for j, widx in enumerate(cand_idxs):
        start_sample = widx * hop_samples
        window = audio[start_sample : start_sample + TARGET_SAMPLES]
        if len(window) < TARGET_SAMPLES:
            continue
        rms = compute_rms(window)
        if rms < RMS_THRESHOLD:
            rejected_rms += 1
            continue
        if np.max(np.abs(window)) < 0.0015:
            rejected_peak += 1
            continue
        # Compute mosquito_score for every candidate that passes gates
        mosquito_ratio, band_energy = mosquito_score(window, sr)
        
        # Update bucket based on mosquito_ratio (ratio-based hard trigger)
        if bucket_by_name == "easy" and mosquito_ratio > 0.20:
            bucket = "hard"
            ratio_hard_file = True  # Set flag for this file
        
        # Soft filter: reject extreme outliers from easy bucket only
        if bucket == "easy" and mosquito_ratio > 0.60:
            rejected_easy_high_mosquito += 1
            continue
        
        peak = np.abs(window).max()
        if peak >= CLIP_THRESHOLD:
            clipping_warnings += 1
        candidates.append((window, rms, widx, mosquito_ratio, band_energy, bucket))
    
    # Track per-file ratio-based hard detection
    if ratio_hard_file:
        stats['hard_files_by_ratio'] += 1
    
    # Track rejection stats globally
    stats['rejected_rms'] += rejected_rms
    stats['rejected_peak'] += rejected_peak
    stats['rejected_easy_high_mosquito'] += rejected_easy_high_mosquito
    if len(candidates) == 0:
        if VERBOSE:
            print(f"  {filepath.name}: 0 candidates (rejected: rms={rejected_rms}, peak={rejected_peak}, easy_high_mosquito={rejected_easy_high_mosquito})")
        return 0
    stats['total_candidates'] += len(candidates)
    candidates.sort(key=lambda x: x[1], reverse=True)
    # Remove impulse spikes (very high crest factor)
    filtered = [c for c in candidates if (np.abs(c[0]).max() / (c[1] + 1e-8)) < 20]
    if len(filtered) >= TOPK_ENERGY:
        candidates = filtered
    
    # Determine per-file random-K cap based on final bucket distribution
    # Use the bucket from first candidate (they should all be from same file anyway)
    file_bucket = candidates[0][5] if candidates else bucket
    RANDOMK_FILE = 60 if file_bucket == "hard" else RANDOMK
    max_to_save = MAX_WINDOWS_PER_FILE
    if global_limit_remaining is not None:
        max_to_save = min(max_to_save, global_limit_remaining)
    # Hard cap per file for resource limits
    max_to_save = min(max_to_save, HARD_CAP_PER_FILE)
    # Select top-K by energy, enforcing minimum distance for diversity
    MIN_DISTANCE = 5
    selected_topk = []
    for candidate in candidates:
        if len(selected_topk) >= min(TOPK_ENERGY, max_to_save):
            break
        widx = candidate[2]
        if all(abs(widx - s[2]) >= MIN_DISTANCE for s in selected_topk):
            selected_topk.append(candidate)
    # Fill remaining top-K slots if needed
    if len(selected_topk) < min(TOPK_ENERGY, len(candidates), max_to_save):
        selected_indices = {s[2] for s in selected_topk}
        for candidate in candidates:
            if candidate[2] not in selected_indices and len(selected_topk) < min(TOPK_ENERGY, max_to_save):
                selected_topk.append(candidate)
                selected_indices.add(candidate[2])
    
    # Get remaining candidates (compare by index to avoid numpy array comparison)
    selected_indices = {s[2] for s in selected_topk}
    remaining_candidates = [c for c in candidates if c[2] not in selected_indices]
    remaining_slots = max_to_save - len(selected_topk)
    if remaining_candidates and remaining_slots > 0:
        randomk = min(RANDOMK_FILE, len(remaining_candidates), remaining_slots)
        # Sample by index positions to avoid numpy array comparison issues
        idxs = list(range(len(remaining_candidates)))
        picked = rng_py.sample(idxs, randomk)
        selected_random = [remaining_candidates[i] for i in picked]
    else:
        selected_random = []
    selected = selected_topk + selected_random
    selected.sort(key=lambda x: x[2])
    windows_saved = 0
    for window, rms, widx, mosquito_ratio, band_energy, bucket in selected:
        if NORMALIZE_WINDOWS:
            peak = np.abs(window).max()
            if peak > 1e-4:
                window = window * (0.95 / peak)
        output_filename = f"Background__{fid}__{file_prefix}__idx{widx:06d}.wav"
        output_path = output_dir / output_filename
        sf.write(output_path, window, TARGET_SR, subtype='PCM_16')
        rel_path = output_path.relative_to(Path('data/processed'))
        stats['manifest'].append({
            'path': str(rel_path).replace('\\', '/'),
            'label': 'Background',
            'source': 'Background',
            'species': 'Background',
            'origin': origin_key,
            'rms': f"{rms:.6f}",
            'bucket': bucket,
            'mosquito_ratio': f"{mosquito_ratio:.6f}",
            'band_energy': f"{band_energy:.2f}"
        })
        # Track mosquito_ratio per bucket for global stats
        if bucket == "hard":
            stats['mosquito_ratios_hard'].append(mosquito_ratio)
        else:
            stats['mosquito_ratios_easy'].append(mosquito_ratio)
        stats['bucket_counts'][bucket] += 1
        windows_saved += 1
    if clipping_warnings > 0:
        stats['clipping_files'].append(f"{filepath.name}: {clipping_warnings} windows near clipping")
    return windows_saved


def preprocess_background():
    """Main entry point for Background preprocessing."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    input_base = Path('data/raw/Background')
    output_base = Path('data/processed/Background')
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
    print(f"  Hop duration:           {HOP_SEC} s (adaptive: 0.1s for <=10s, 0.25s for <=30s)")
    print(f"  Max candidates per file: {MAX_CANDIDATES_PER_FILE}")
    print(f"  RMS threshold:          {RMS_THRESHOLD} (permissive for quiet ambience)")
    print(f"  Top-K by energy:        {TOPK_ENERGY} (captures events)")
    print(f"  Random-K from rest:     {RANDOMK} (captures ambience)")
    print(f"  Max windows per file:   {MAX_WINDOWS_PER_FILE}")
    print(f"  Hard cap per file:      {HARD_CAP_PER_FILE}")
    print(f"  Global max windows:     {GLOBAL_MAX_WINDOWS if GLOBAL_MAX_WINDOWS else 'Unlimited (clamped per file)'}")
    print(f"  Normalize windows:      {NORMALIZE_WINDOWS}")
    print(f"  Clipping threshold:     {CLIP_THRESHOLD}")
    print(f"  Random seed:            {RANDOM_SEED}")
    print("="*80 + "\n")
    output_base.mkdir(parents=True, exist_ok=True)
    # Remove any existing output to avoid duplicates
    for wav_file in output_base.rglob('*.wav'):
        wav_file.unlink()
    stats = {
        'windows_created': 0,
        'files_processed': 0,
        'total_candidates': 0,
        'rejected_rms': 0,
        'rejected_peak': 0,
        'rejected_easy_high_mosquito': 0,
        'hard_files_by_name': 0,
        'hard_files_by_ratio': 0,
        'errors': [],
        'clipping_files': [],
        'manifest': [],
        'mosquito_ratios_hard': [],
        'mosquito_ratios_easy': [],
        'bucket_counts': {'hard': 0, 'easy': 0}
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
    print(f"  Files that produced windows: {stats['files_processed']}")
    print(f"  Total files scanned:    {len(audio_files)}")
    if stats['files_processed'] > 0:
        avg_windows_per_file = stats['windows_created'] / stats['files_processed']
        print(f"  Avg windows per file:   {avg_windows_per_file:.1f}")
    print(f"\nRejection stats:")
    print(f"  Rejected (RMS too low): {stats['rejected_rms']}")
    print(f"  Rejected (peak too low): {stats['rejected_peak']}")
    print(f"  Rejected (easy high mosquito >0.60): {stats['rejected_easy_high_mosquito']}")
    total_rejected = stats['rejected_rms'] + stats['rejected_peak'] + stats['rejected_easy_high_mosquito']
    total_scanned = total_rejected + stats['total_candidates']
    if total_scanned > 0:
        print(f"  Total windows scanned:  {total_scanned}")
        print(f"  Candidate acceptance:   {100.0 * stats['total_candidates'] / total_scanned:.1f}% (passed gates)")
        print(f"  Final save rate:        {100.0 * stats['windows_created'] / total_scanned:.1f}% (actually saved)")
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
    
    # Bucket summary
    print(f"\nBucket distribution:")
    total_saved = stats['bucket_counts']['hard'] + stats['bucket_counts']['easy']
    if total_saved > 0:
        hard_pct = 100.0 * stats['bucket_counts']['hard'] / total_saved
        easy_pct = 100.0 * stats['bucket_counts']['easy'] / total_saved
        print(f"  Hard negatives:         {stats['bucket_counts']['hard']} ({hard_pct:.1f}%)")
        print(f"  Easy background:        {stats['bucket_counts']['easy']} ({easy_pct:.1f}%)")
    print(f"\nBucket assignment triggers:")
    print(f"  Hard files by filename:       {stats['hard_files_by_name']}")
    print(f"  Hard files by ratio (>0.20):  {stats['hard_files_by_ratio']}")
    
    # Mosquito score percentile stats per bucket
    print(f"\nMosquito score statistics by bucket:")
    for bucket_name in ['hard', 'easy']:
        ratios = stats[f'mosquito_ratios_{bucket_name}']
        if ratios:
            ratios_arr = np.array(ratios)
            print(f"  {bucket_name.upper()}:")
            print(f"    Mean:   {ratios_arr.mean():.6f}")
            print(f"    Median: {np.median(ratios_arr):.6f}")
            print(f"    P90:    {np.percentile(ratios_arr, 90):.6f}")
            print(f"    Max:    {ratios_arr.max():.6f}")
            if bucket_name == 'easy':
                print(f"    (Note: Max capped at 0.60 by soft filter)")
        else:
            print(f"  {bucket_name.upper()}: No windows saved")
    
    print("="*80)
    print("\nPreprocessing complete!")
    print(f"Output saved to: {output_base.absolute()}")
    # Write manifest CSV
    manifest_path = Path('data/splits/background_manifest.csv')
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if stats['manifest']:
        with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['path', 'label', 'source', 'species', 'origin', 'rms', 'bucket', 'mosquito_ratio', 'band_energy']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(stats['manifest'])
        print(f"Manifest saved to: {manifest_path.absolute()}")
    print()


def main():
    preprocess_background()


if __name__ == "__main__":
    main()


