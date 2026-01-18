"""
Preprocess Wingsbeats dataset: standardize each micro-clip to 1-second windows for ML training.
Usage:
    python scripts/preprocess_wingsbeats.py
"""

from pathlib import Path
from collections import defaultdict
import warnings
import csv
import hashlib
import numpy as np
import soundfile as sf
from tqdm import tqdm

try:
    import librosa
except ImportError:
    librosa = None

warnings.filterwarnings('ignore')

# Constants
TARGET_SR = 16000
TARGET_DURATION = 1.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)

CLASS_MAPPING = {
    'Aedes aegypti': 'Aedes',
    'Aedes albopictus': 'Aedes',
    'Anopheles arabiensis': 'Anopheles',
    'Anopheles gambiae': 'Anopheles',
    'Culex pipiens': 'Culex',
    'Culex quinquefasciatus': 'Culex',
}

CLIP_RMS_THRESHOLD = 0.0002
MIN_CLIP_SAMPLES = 200
MAX_WINDOWS_PER_RECORDING = 200


def load_audio_smart(filepath, target_sr=TARGET_SR):
    """Robust audio loader: mono conversion and resampling if needed."""
    try:
        audio, sr = sf.read(filepath, dtype='float32')
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            if librosa is None:
                raise RuntimeError("librosa required for resampling but not installed")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return audio, sr
    except Exception as e:
        raise RuntimeError(f"Failed to load {filepath}: {e}")


def get_recording_folders(species_dir):
    """Find leaf D_* folders with WAVs (no nested D_*)."""
    recording_folders = []
    for item in species_dir.rglob('*'):
        if item.is_dir() and item.name.startswith('D_') and len(item.name.split('_')) >= 7:
            has_wavs = any(item.rglob('*.wav'))
            has_nested_d = any(
                sub.is_dir() and sub.name.startswith('D_') and len(sub.name.split('_')) >= 7
                for sub in item.rglob('*')
            )
            if has_wavs and not has_nested_d:
                recording_folders.append(item)
    return sorted(recording_folders)


def collect_wav_files(recording_folder):
    """List all .wav files in a folder."""
    return sorted([f for f in recording_folder.rglob('*.wav') if f.is_file()])


def standardize_to_1s(audio, target_samples=TARGET_SAMPLES, rng=None):
    """Crop or tile audio to exactly target_samples."""
    n = len(audio)
    if n == 0:
        return None

    if n >= target_samples:
        # Crop (center or random crop)
        if rng is not None:
            start = rng.randint(0, n - target_samples + 1)
        else:
            start = (n - target_samples) // 2
        return audio[start:start + target_samples].astype(np.float32)

    # Tile/loop short clips to avoid zero-padding artifacts
    num_repeats = (target_samples // n) + 1
    tiled = np.tile(audio, num_repeats)[:target_samples]
    
    # Apply random circular shift for augmentation (short clips only)
    if rng is not None:
        shift = rng.randint(0, target_samples)
        tiled = np.roll(tiled, shift)
    
    return tiled.astype(np.float32)


def process_recording_folder(recording_folder, species_name, class_name, output_dir, input_base, stats):
    """
    For each micro-clip: resample, pad/crop to 1s, filter, and save as a single window.
    No merging across clips.
    """
    recording_name = recording_folder.name
    wav_files = collect_wav_files(recording_folder)
    if len(wav_files) == 0:
        return 0, 0, 0, 0, False

    seed_key = str(recording_folder.relative_to(input_base)).replace('\\', '/')
    folder_seed = int(hashlib.md5(seed_key.encode('utf-8')).hexdigest()[:8], 16)

    window_count = 0
    clips_processed_local = 0
    clips_skipped_local = 0
    total_clips_seen = 0

    for i, wav_file in enumerate(wav_files):
        if window_count >= MAX_WINDOWS_PER_RECORDING:
            break

        total_clips_seen += 1
        try:
            audio, sr = load_audio_smart(wav_file, target_sr=TARGET_SR)
            audio = audio.astype(np.float32, copy=False)
            
            # Remove DC offset (some devices have tiny bias)
            audio = audio - np.mean(audio)

            if audio is None or len(audio) < MIN_CLIP_SAMPLES:
                clips_skipped_local += 1
                continue

            # Minimal quality check: skip low-RMS clips
            clip_rms = float(np.sqrt(np.mean(audio ** 2)))
            if clip_rms < CLIP_RMS_THRESHOLD:
                clips_skipped_local += 1
                continue

            # Standardize to 1s window (tile or crop, with augmentation)
            window = standardize_to_1s(
                audio,
                target_samples=TARGET_SAMPLES,
                rng=np.random.RandomState(folder_seed + i)
            )
            if window is None:
                clips_skipped_local += 1
                continue

            # Peak normalization for stable training (headroom 0.95)
            peak = np.abs(window).max()
            if peak > 1e-4:
                window = window * (0.95 / peak)

            # Compute final RMS for manifest
            win_rms = float(np.sqrt(np.mean(window ** 2)))

            # Save window
            output_filename = f"{species_name.replace(' ', '_')}__{recording_name}__clip{window_count:05d}.wav"
            output_path = output_dir / output_filename
            sf.write(output_path, window, TARGET_SR, subtype='PCM_16')

            rel_path = output_path.relative_to(Path('data/processed'))
            recording_id = f"{species_name}__{recording_name}"
            window_id = f"{recording_id}__clip{window_count:05d}"

            stats['manifest'].append({
                'path': str(rel_path).replace('\\', '/'),
                'label': class_name,
                'source': 'Wingsbeats',
                'species': species_name,
                'recording': recording_name,
                'recording_id': recording_id,
                'window_id': window_id,
                'origin': seed_key,
                'rms': f"{win_rms:.6f}"
            })

            window_count += 1
            clips_processed_local += 1

        except Exception as e:
            stats['errors'].append(f"{wav_file.name}: {str(e)}")
            clips_skipped_local += 1
            continue

    recording_used = window_count > 0
    return window_count, clips_processed_local, clips_skipped_local, total_clips_seen, recording_used


def preprocess_wingsbeats():
    """Main entry point for Wingsbeats preprocessing."""
    input_base = Path('data/raw/Wingsbeats')
    output_base = Path('data/processed/Wingsbeats')
    if not input_base.exists():
        print(f"ERROR: Input directory not found: {input_base}")
        return
    # Remove existing output to avoid duplicates on reruns
    if output_base.exists():
        for wav_file in output_base.rglob('*.wav'):
            wav_file.unlink()
    print("="*80)
    print(" "*25 + "WINGSBEATS PREPROCESSING")
    print("="*80)
    print(f"Input:  {input_base.absolute()}")
    print(f"Output: {output_base.absolute()}")
    print(f"\nConfiguration:")
    print(f"  Target sample rate:     {TARGET_SR} Hz")
    print(f"  Target duration:        {TARGET_DURATION} s ({TARGET_SAMPLES} samples)")
    print(f"  Clip RMS threshold:     {CLIP_RMS_THRESHOLD} (skip clips below this)")
    print(f"  Min clip length:        {MIN_CLIP_SAMPLES} samples (~{MIN_CLIP_SAMPLES/TARGET_SR*1000:.1f} ms)")
    print(f"  Short-clip handling:    Loop/tile (no zero-padding)")
    print(f"  Augmentation:           Random circular shift (short clips) + random crop (long clips)")
    print(f"  Normalization:          Peak normalize to 0.95 headroom")
    print(f"  Max windows/recording:  {MAX_WINDOWS_PER_RECORDING}")
    print("="*80 + "\n")
    stats = {
        'windows_created': defaultdict(int),
        'recordings_processed': defaultdict(int),
        'clips_skipped': 0,
        'clips_processed': 0,
        'total_clips_seen': 0,
        'errors': [],
        'manifest': []
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
        recording_folders = get_recording_folders(species_dir)
        if len(recording_folders) == 0:
            print(f"WARNING: No recording folders found for {species_name}")
            continue
        print(f"Processing {species_name} -> {class_name}")
        print(f"  Found {len(recording_folders)} recording folders")
        species_windows = 0
        species_clips_processed = 0
        species_clips_skipped = 0
        species_clips_seen = 0
        species_recordings_used = 0
        for recording_folder in tqdm(recording_folders, desc=f"  {species_name}", ncols=100):
            windows, clips_proc, clips_skip, clips_seen, recording_used = process_recording_folder(
                recording_folder=recording_folder,
                species_name=species_name,
                class_name=class_name,
                output_dir=output_dir,
                input_base=input_base,
                stats=stats
            )
            species_windows += windows
            species_clips_processed += clips_proc
            species_clips_skipped += clips_skip
            species_clips_seen += clips_seen
            if recording_used:
                species_recordings_used += 1
        stats['windows_created'][class_name] += species_windows
        stats['recordings_processed'][class_name] += species_recordings_used
        stats['clips_processed'] += species_clips_processed
        stats['clips_skipped'] += species_clips_skipped
        stats['total_clips_seen'] += species_clips_seen
        acceptance_rate = (species_clips_processed / species_clips_seen * 100) if species_clips_seen > 0 else 0
        print(f"  Created {species_windows} windows from {species_recordings_used}/{len(recording_folders)} recordings" +
              f" ({species_clips_processed} clips processed, {species_clips_skipped} skipped)")
        print(f"  Clip acceptance rate: {acceptance_rate:.1f}% ({species_clips_processed}/{species_clips_seen})\n")
    print("\n" + "="*80)
    print(" "*30 + "SUMMARY STATISTICS")
    print("="*80)
    total_windows = sum(stats['windows_created'].values())
    total_recordings = sum(stats['recordings_processed'].values())
    print("\nWindows created by class:")
    for class_name in sorted(stats['windows_created'].keys()):
        count = stats['windows_created'][class_name]
        pct = (count / total_windows * 100) if total_windows > 0 else 0
        print(f"  {class_name:15s} {count:6d} windows ({pct:5.1f}%)")
    print(f"\nTotal:")
    print(f"  Windows created:        {total_windows}")
    print(f"  Recordings processed:   {total_recordings}")
    print(f"  Micro-clips processed:  {stats['clips_processed']}")
    print(f"  Micro-clips skipped:    {stats['clips_skipped']} (short or low RMS)")
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
    manifest_path = Path('data/splits/wingsbeats_manifest.csv')
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if stats['manifest']:
        with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['path', 'label', 'source', 'species', 'recording', 'recording_id', 'window_id', 'origin', 'rms']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(stats['manifest'])
        print(f"Manifest saved to: {manifest_path.absolute()}")
    print()


def main():
    preprocess_wingsbeats()

if __name__ == "__main__":
    main()

