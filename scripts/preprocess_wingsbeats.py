"""
Preprocess Wingsbeats dataset: merge micro-clips into 1-second standardized windows.

Usage:
    python scripts/preprocess_wingsbeats.py
"""

from pathlib import Path
from collections import defaultdict
import warnings
import csv
import numpy as np
import soundfile as sf
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Constants
TARGET_SR = 16000
TARGET_DURATION = 1.0  # seconds
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)

# Class mapping: species -> class name
CLASS_MAPPING = {
    'Aedes aegypti': 'Aedes',
    'Aedes albopictus': 'Aedes',
    'Anopheles arabiensis': 'Anopheles',
    'Anopheles gambiae': 'Anopheles',
    'Culex pipiens': 'Culex',
    'Culex quinquefasciatus': 'Culex',
}

# Configuration
PEAK_THRESHOLD = 0.01  # Skip clips with peak below this (more reliable for tiny bursts)
MIN_CLIP_SAMPLES = 200  # Minimum clip length (~12.5ms at 16kHz) to avoid unstable RMS
FADE_MS = 3  # Fade duration in ms (prevents click artifacts)
FADE_SAMPLES = int(TARGET_SR * FADE_MS / 1000)
GAP_MS = 0  # Silence gap between clips in milliseconds
MIN_TAIL_SEC = 0.3  # Minimum tail length to keep and pad
MAX_WINDOWS_PER_RECORDING = 200  # Cap windows per recording folder
NORMALIZE_CLIPS = False  # Don't normalize clips (can amplify junk)
SILENCE_RATIO_THRESHOLD = 0.95  # Drop windows with >95% near-silence


def normalize_clip(audio, headroom=0.95):
    """Peak normalize audio clip to prevent clipping."""
    peak = np.abs(audio).max()
    if peak > 1e-6:  # Avoid division by zero
        return audio * (headroom / peak)
    return audio


def apply_fade(audio, fade_samples=FADE_SAMPLES):
    """Apply fade-in/out to prevent click artifacts when concatenating."""
    if len(audio) <= 2 * fade_samples:
        return audio
    w = np.ones(len(audio), dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    w[:fade_samples] = ramp
    w[-fade_samples:] = ramp[::-1]
    return audio * w


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
        # Try loading with soundfile first
        audio, sr = sf.read(filepath, dtype='float32')
        
        # Convert to mono if needed
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        
        # Resample if needed
        if sr != target_sr:
            # Import librosa only when resampling is needed
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        return audio, sr
    
    except Exception as e:
        raise RuntimeError(f"Failed to load {filepath}: {e}")


def get_recording_folders(species_dir):
    """
    Find all recording folders (D_*) in a species directory.
    
    Returns:
        List of Path objects to recording folders
    """
    recording_folders = []
    
    for item in species_dir.rglob('*'):
        if item.is_dir() and item.name.startswith('D_') and len(item.name.split('_')) >= 7:
            recording_folders.append(item)
    
    return sorted(recording_folders)


def collect_wav_files(recording_folder):
    """Collect all .wav files in recording folder."""
    return sorted([f for f in recording_folder.rglob('*.wav') if f.is_file()])


def process_recording_folder(recording_folder, species_name, class_name, output_dir, stats):
    """
    Process one recording folder: merge micro-clips into 1-second windows.
    
    Args:
        recording_folder: Path to recording folder
        species_name: Original species name
        class_name: Output class name (Aedes/Anopheles/Culex)
        output_dir: Output directory for this class
        stats: Dictionary to accumulate statistics
    
    Returns:
        Tuple of (windows_created, clips_processed, clips_skipped, recording_used)
    """
    recording_name = recording_folder.name
    wav_files = collect_wav_files(recording_folder)
    
    if len(wav_files) == 0:
        return 0, 0, 0, False
    
    # Use list to accumulate clips for better performance
    clip_list = []
    remainder = np.array([], dtype=np.float32)  # Track remainder separately
    gap_samples = int((GAP_MS / 1000.0) * TARGET_SR)
    silence_gap = np.zeros(gap_samples, dtype=np.float32) if gap_samples > 0 else None
    
    window_count = 0
    clips_processed_local = 0
    clips_skipped_local = 0
    error_count = 0
    
    for wav_file in wav_files:
        if window_count >= MAX_WINDOWS_PER_RECORDING:
            break
        
        try:
            audio, sr = load_audio_smart(wav_file, target_sr=TARGET_SR)
            
            if len(audio) < MIN_CLIP_SAMPLES:
                clips_skipped_local += 1
                continue
            
            if np.max(np.abs(audio)) < PEAK_THRESHOLD:
                clips_skipped_local += 1
                continue
            
            if NORMALIZE_CLIPS:
                audio = normalize_clip(audio)
            
            audio = apply_fade(audio)
            
            if silence_gap is not None and clips_processed_local > 0:
                clip_list.append(silence_gap)
            clip_list.append(audio)
            clips_processed_local += 1
            
            # Batch every 50 clips to avoid O(n²) concat
            if len(clip_list) >= 50:
                if len(remainder) > 0:
                    buffer = np.concatenate([remainder] + clip_list)
                else:
                    buffer = np.concatenate(clip_list)
                clip_list = []
                
                while len(buffer) >= TARGET_SAMPLES and window_count < MAX_WINDOWS_PER_RECORDING:
                    window = buffer[:TARGET_SAMPLES]
                    buffer = buffer[TARGET_SAMPLES:]
                    
                    # Window-level RMS gate
                    win_rms = np.sqrt(np.mean(window**2))
                    if win_rms < 0.001:
                        continue
                    
                    silence_ratio = (np.abs(window) < 1e-4).mean()
                    if silence_ratio > SILENCE_RATIO_THRESHOLD:
                        continue
                    
                    output_filename = f"{species_name.replace(' ', '_')}__{recording_name}__win{window_count:04d}.wav"
                    output_path = output_dir / output_filename
                    sf.write(output_path, window, TARGET_SR, subtype='PCM_16')
                    
                    # Record in manifest
                    rel_path = output_path.relative_to(Path('data/processed'))
                    stats['manifest'].append({
                        'path': str(rel_path).replace('\\', '/'),
                        'label': class_name,
                        'source': 'Wingsbeats',
                        'species': species_name,
                        'origin': recording_name,
                        'rms': f"{win_rms:.6f}"
                    })
                    
                    window_count += 1
                
                remainder = buffer
        
        except Exception as e:
            error_count += 1
            stats['errors'].append(f"{wav_file.name}: {str(e)}")
            continue
    
    # Final batch
    if len(clip_list) > 0:
        buffer = np.concatenate([remainder] + clip_list) if len(remainder) > 0 else np.concatenate(clip_list)
    else:
        buffer = remainder
    
    while len(buffer) >= TARGET_SAMPLES and window_count < MAX_WINDOWS_PER_RECORDING:
        window = buffer[:TARGET_SAMPLES]
        buffer = buffer[TARGET_SAMPLES:]
        
        # Window-level RMS gate
        win_rms = np.sqrt(np.mean(window**2))
        if win_rms < 0.001:
            continue
        
        silence_ratio = (np.abs(window) < 1e-4).mean()
        if silence_ratio > SILENCE_RATIO_THRESHOLD:
            continue
        
        output_filename = f"{species_name.replace(' ', '_')}__{recording_name}__win{window_count:04d}.wav"
        output_path = output_dir / output_filename
        sf.write(output_path, window, TARGET_SR, subtype='PCM_16')
        
        # Record in manifest
        rel_path = output_path.relative_to(Path('data/processed'))
        stats['manifest'].append({
            'path': str(rel_path).replace('\\', '/'),
            'label': class_name,
            'source': 'Wingsbeats',
            'species': species_name,
            'origin': recording_name,
            'rms': f"{win_rms:.6f}"
        })
        
        window_count += 1
    
    # Pad and save tail if long enough
    if len(buffer) >= int(MIN_TAIL_SEC * TARGET_SR) and window_count < MAX_WINDOWS_PER_RECORDING:
        padding_needed = TARGET_SAMPLES - len(buffer)
        buffer = np.concatenate([buffer, np.zeros(padding_needed, dtype=np.float32)]) if padding_needed > 0 else buffer[:TARGET_SAMPLES]
        
        # Window-level RMS gate
        win_rms = np.sqrt(np.mean(buffer**2))
        if win_rms >= 0.001:
            silence_ratio = (np.abs(buffer) < 1e-4).mean()
            if silence_ratio <= SILENCE_RATIO_THRESHOLD:
                output_filename = f"{species_name.replace(' ', '_')}__{recording_name}__win{window_count:04d}.wav"
                output_path = output_dir / output_filename
                sf.write(output_path, buffer, TARGET_SR, subtype='PCM_16')
                
                # Record in manifest
                rel_path = output_path.relative_to(Path('data/processed'))
                stats['manifest'].append({
                    'path': str(rel_path).replace('\\', '/'),
                    'label': class_name,
                    'source': 'Wingsbeats',
                    'species': species_name,
                    'origin': recording_name,
                    'rms': f"{win_rms:.6f}"
                })
                
                window_count += 1
    
    recording_used = window_count > 0
    
    return window_count, clips_processed_local, clips_skipped_local, recording_used


def preprocess_wingsbeats():
    """Main preprocessing for Wingsbeats dataset."""
    input_base = Path('data/raw/Wingsbeats')
    output_base = Path('data/processed/Wingsbeats')
    
    if not input_base.exists():
        print(f"ERROR: Input directory not found: {input_base}")
        return
    
    print("="*80)
    print(" "*25 + "WINGSBEATS PREPROCESSING")
    print("="*80)
    print(f"Input:  {input_base.absolute()}")
    print(f"Output: {output_base.absolute()}")
    print(f"\nConfiguration:")
    print(f"  Target sample rate:     {TARGET_SR} Hz")
    print(f"  Target duration:        {TARGET_DURATION} s ({TARGET_SAMPLES} samples)")
    print(f"  Peak threshold:         {PEAK_THRESHOLD} (skip clips below this)")
    print(f"  Min clip length:        {MIN_CLIP_SAMPLES} samples (~{MIN_CLIP_SAMPLES/TARGET_SR*1000:.1f} ms)")
    print(f"  Fade duration:          {FADE_MS} ms (prevents clicks)")
    print(f"  Normalize clips:        {NORMALIZE_CLIPS}")
    print(f"  Silence ratio filter:   {SILENCE_RATIO_THRESHOLD} (drop mostly-silent windows)")
    print(f"  Gap between clips:      {GAP_MS} ms")
    print(f"  Min tail length:        {MIN_TAIL_SEC} s")
    print(f"  Max windows/recording:  {MAX_WINDOWS_PER_RECORDING}")
    print("="*80 + "\n")
    
    stats = {
        'windows_created': defaultdict(int),
        'recordings_processed': defaultdict(int),
        'clips_skipped': 0,
        'clips_processed': 0,
        'errors': [],
        'manifest': []  # CSV records
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
        species_recordings_used = 0
        

        for recording_folder in tqdm(recording_folders, desc=f"  {species_name}", ncols=100):
            windows, clips_proc, clips_skip, recording_used = process_recording_folder(
                recording_folder=recording_folder,
                species_name=species_name,
                class_name=class_name,
                output_dir=output_dir,
                stats=stats
            )
            species_windows += windows
            species_clips_processed += clips_proc
            species_clips_skipped += clips_skip
            if recording_used:
                species_recordings_used += 1
        
        stats['windows_created'][class_name] += species_windows
        stats['recordings_processed'][class_name] += species_recordings_used
        stats['clips_processed'] += species_clips_processed
        stats['clips_skipped'] += species_clips_skipped
        
        print(f"  Created {species_windows} windows from {species_recordings_used}/{len(recording_folders)} recordings" +
              f" ({species_clips_processed} clips processed, {species_clips_skipped} skipped)\n")
    
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
    
    # Write manifest CSV
    manifest_path = Path('data/splits/wingsbeats_manifest.csv')
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
    preprocess_wingsbeats()


if __name__ == "__main__":
    main()


