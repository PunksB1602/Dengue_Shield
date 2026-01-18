"""
Sanity check for preprocessed datasets.
Usage:
    python scripts/sanity_check_processed.py Wingsbeats
    python scripts/sanity_check_processed.py Abuzz
    python scripts/sanity_check_processed.py Background
"""

import sys
from pathlib import Path
import random
import hashlib
import csv
import numpy as np
import soundfile as sf
from collections import defaultdict

# Audio parameters
TARGET_SR = 16000
TARGET_DURATION = 1.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)
SAMPLES_PER_CLASS = 10

# Mosquito-band parameters (matches preprocess scripts)
FMAX = 4000
BAND_LO, BAND_HI = 250, 1200
MOSQUITO_SCORE_THRESHOLD_MIN = 0.02
MOSQUITO_SCORE_THRESHOLD_MAX = 0.12


def count_files_per_class(processed_dir):
    """Count WAV files in each class directory."""
    class_counts = {}
    for class_dir in sorted(processed_dir.iterdir()):
        if class_dir.is_dir():
            wav_files = list(class_dir.glob('*.wav'))
            class_counts[class_dir.name] = len(wav_files)
    return class_counts


def mosquito_score(window, sr=TARGET_SR):
    """Mosquito-band energy ratio (250-1200 Hz, Hann window)."""
    x = window.astype(np.float32)
    x = x - np.mean(x)
    w = np.hanning(len(x)).astype(np.float32)
    spec = np.abs(np.fft.rfft(x * w))**2
    freqs = np.fft.rfftfreq(len(x), d=1.0/sr)
    mask_all = (freqs >= 0) & (freqs <= FMAX)
    mask_band = (freqs >= BAND_LO) & (freqs <= BAND_HI)
    return float(spec[mask_band].sum() / (spec[mask_all].sum() + 1e-12))


def check_file_duration_and_content(filepath, check_mode='rms_only', dataset_name=''):
    """Check file format and content based on dataset type."""
    try:
        audio, sr = sf.read(filepath, dtype='float32', always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        sr_ok = (sr == TARGET_SR)
        samples_ok = (len(audio) == TARGET_SAMPLES)
        format_ok = sr_ok and samples_ok
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.abs(audio).max()
        m_score = mosquito_score(audio, sr=sr)  # Compute for stats, but don't use for pass/fail
        
        # Dataset-aware content thresholds
        if dataset_name.lower() == 'wingsbeats':
            has_content = (rms > 0.0002) and (peak > 0.003)
        elif dataset_name.lower() == 'abuzz':
            has_content = (rms > 0.0005) and (peak > 0.005)
        elif dataset_name.lower() == 'background':
            has_content = (rms > 0.00025) or (peak > 0.0015)
        else:
            # Fallback based on check_mode for backward compatibility
            if check_mode == 'mosquito_positive':
                has_content = (rms > 0.0005) and (peak > 0.003)
            elif check_mode == 'mosquito_negative':
                has_content = (rms > 0.0007) or (peak > 0.003)
            else:
                has_content = rms > 0.001
        return format_ok, has_content, sr, len(audio), rms, peak, m_score
    except Exception:
        return False, False, 0, 0, 0.0, 0.0, None


def play_audio_sample(filepath):
    """Print audio file info; playback optional (see code)."""
    try:
        audio, sr = sf.read(filepath, dtype='float32', always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        duration = len(audio) / sr
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.abs(audio).max()
        print(f"    {filepath.name}")
        print(f"      Duration: {duration:.3f}s | RMS: {rms:.4f} | Peak: {peak:.4f}")
        # To enable playback, install sounddevice and uncomment below:
        # import sounddevice as sd
        # sd.play(audio, sr)
        # sd.wait()
    except Exception as e:
        print(f"    ERROR: {filepath.name} - {str(e)}")


def sanity_check():
    """Run sanity checks on processed dataset."""
    DATASET = sys.argv[1] if len(sys.argv) > 1 else "Abuzz"
    # Try common path variants for robustness
    candidates = [
        Path(f"data/processed/{DATASET}"),
        Path(f"data/processed/{DATASET.lower()}"),
        Path(f"data/processed/{DATASET.capitalize()}"),
    ]
    processed_dir = None
    if DATASET.lower() == "background":
        processed_dir = Path("data/processed/Background")
    else:
        for p in candidates:
            if p.exists():
                processed_dir = p
                break
        if processed_dir is None:
            processed_dir = candidates[0]
    # Select check mode
    if DATASET.lower() in ["abuzz", "wingsbeats"]:
        check_mode = 'mosquito_positive'
    elif DATASET.lower() == "background":
        check_mode = 'mosquito_negative'
    else:
        check_mode = 'rms_only'
    if not processed_dir.exists():
        print(f"ERROR: Processed directory not found: {processed_dir}")
        print(f"Please run preprocess_{DATASET.lower()}.py first.")
        return
    print("="*80)
    print(" "*25 + f"SANITY CHECK - {DATASET.upper()}")
    print("="*80)
    print(f"Checking: {processed_dir.absolute()}")
    if check_mode == 'mosquito_positive':
        print(f"Content check: RMS > 0.0005 and peak > 0.003 (matches preprocessing)")
        print(f"Mosquito score computed for stats only (band: {BAND_LO}-{BAND_HI} Hz)\n")
    elif check_mode == 'mosquito_negative':
        print(f"Content check: RMS > 0.0007 or peak > 0.003 (quiet ambience allowed)")
        print(f"Mosquito score computed for stats only (band: {BAND_LO}-{BAND_HI} Hz)\n")
    else:
        print(f"Using RMS content check only\n")
    # Check 1: File counts per class
    print("="*80)
    print("CHECK 1: File counts per class")
    print("="*80)
    if DATASET.lower() == "background":
        wav_files = list(processed_dir.glob('*.wav'))
        class_counts = {"Background": len(wav_files)}
    else:
        class_counts = count_files_per_class(processed_dir)
    if len(class_counts) == 0:
        print("ERROR: No class directories found!")
        return
    total_files = sum(class_counts.values())
    print("\nFiles per class:")
    for class_name, count in sorted(class_counts.items()):
        pct = (count / total_files * 100) if total_files > 0 else 0
        print(f"  {class_name:15s} {count:6d} files ({pct:5.1f}%)")
    print(f"\nTotal files: {total_files}")
    # Check for class balance
    if len(class_counts) > 1:
        counts_list = list(class_counts.values())
        max_count = max(counts_list)
        min_count = min(counts_list)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}x")
        if imbalance_ratio > 5.0:
            print("  WARNING: Classes are significantly imbalanced!")
        elif imbalance_ratio > 2.0:
            print("  Classes are somewhat imbalanced (acceptable for now)")
        else:
            print("  Classes are reasonably balanced")
    # Check 2: Random sample verification
    print("\n" + "="*80)
    print("CHECK 2: Random sample verification")
    print("="*80)
    all_issues = []
    class_stats = {}
    audio_hashes = {}  # For duplicate detection
    duplicate_count = 0
    for class_name, count in sorted(class_counts.items()):
        if DATASET.lower() == "background":
            class_dir = processed_dir
            wav_files = list(processed_dir.glob('*.wav'))
        else:
            class_dir = processed_dir / class_name
            wav_files = list(class_dir.glob('*.wav'))
        if len(wav_files) == 0:
            print(f"\n{class_name}: No files found!")
            continue
        num_samples = min(SAMPLES_PER_CLASS, len(wav_files))
        sampled_files = random.sample(wav_files, num_samples)
        print(f"\n{class_name}: Checking {num_samples} random samples")
        format_issues = 0
        content_issues = 0
        peak_values = []
        rms_values = []
        score_values = []
        for filepath in sampled_files:
            format_ok, has_content, sr, num_samples_file, rms, peak, m_score = check_file_duration_and_content(filepath, check_mode, DATASET)
            peak_values.append(peak)
            rms_values.append(rms)
            if m_score is not None:
                score_values.append(m_score)
            # Duplicate check: stable hash using MD5 with RMS+peak for extra reliability
            try:
                audio, _ = sf.read(filepath, dtype='float32', always_2d=False)
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                sig = np.round(audio[::50], 4).astype(np.float32).tobytes()
                meta = f"{rms:.6f}_{peak:.6f}".encode()
                audio_hash = hashlib.md5(sig + meta).hexdigest()
                if audio_hash in audio_hashes:
                    duplicate_count += 1
                    all_issues.append(f"{class_name}/{filepath.name}: Duplicate of {audio_hashes[audio_hash]}")
                else:
                    audio_hashes[audio_hash] = f"{class_name}/{filepath.name}"
            except Exception:
                pass
            if not format_ok:
                format_issues += 1
                all_issues.append(f"{class_name}/{filepath.name}: Wrong format (SR={sr}, samples={num_samples_file})")
            if not has_content:
                content_issues += 1
                if m_score is not None:
                    all_issues.append(f"{class_name}/{filepath.name}: low content (rms={rms:.6f}, peak={peak:.4f}, mscore={m_score:.4f})")
                else:
                    all_issues.append(f"{class_name}/{filepath.name}: low RMS (rms={rms:.6f})")
        class_stats[class_name] = {
            'rms': rms_values,
            'peak': peak_values,
            'score': score_values
        }
        if format_issues > 0:
            print(f" {format_issues}/{num_samples} files have incorrect format (should be 16kHz, 16000 samples)")
        else:
            print(f" All samples have correct format (16kHz, 16000 samples = 1.0s)")
        if content_issues > 0:
            print(f" {content_issues}/{num_samples} files are silent or very low energy")
        else:
            print(f" All samples have audible content")
        if len(peak_values) > 0:
            avg_peak = np.mean(peak_values)
            high_peak_count = sum(1 for p in peak_values if p > 0.9)
            if high_peak_count > num_samples * 0.8:
                print(f" {high_peak_count}/{num_samples} files have peak > 0.9 (avg peak: {avg_peak:.3f})")
                print(f" This suggests aggressive normalization. Check if this is expected.")
            else:
                print(f" Peak levels reasonable (avg: {avg_peak:.3f})")
    # Duplicate summary
    if duplicate_count > 0:
        print(f"\n DUPLICATES DETECTED: {duplicate_count} exact duplicate files found in random sample!")
        print(f" Note: Hash-based detection only catches identical audio. Overlapped windows from")
        print(f" sliding-window extraction (e.g., Abuzz hop=0.5s) are expected and not duplicates.")
        print(f" True duplicates suggest rerun/collision issues. Check 'origin' field in manifest.")
    # Quick stats per class
    print("\n" + "="*80)
    print(" "*25 + "QUICK STATS PER CLASS")
    print("="*80)
    for class_name in sorted(class_stats.keys()):
        stats = class_stats[class_name]
        print(f"\n{class_name}:")
        if stats['rms']:
            rms_arr = np.array(stats['rms'])
            print(f"  RMS:   mean={rms_arr.mean():.4f}, median={np.median(rms_arr):.4f}, std={rms_arr.std():.4f}")
        if stats['peak']:
            peak_arr = np.array(stats['peak'])
            clipping_pct = (peak_arr > 0.95).mean() * 100
            print(f"  Peak:  mean={peak_arr.mean():.3f}, median={np.median(peak_arr):.3f}")
            print(f"  Clipping: {clipping_pct:.1f}% samples > 0.95")
        if stats['score']:
            score_arr = np.array(stats['score'])
            print(f"  Mosquito score: mean={score_arr.mean():.4f}, median={np.median(score_arr):.4f}, min={score_arr.min():.4f}, max={score_arr.max():.4f}")
    
    # Bucket-aware stats for Background dataset
    if DATASET.lower() == "background":
        print("\n" + "="*80)
        print(" "*20 + "BACKGROUND BUCKET ANALYSIS")
        print("="*80)
        manifest_path = Path("data/splits/background_manifest.csv")
        if manifest_path.exists():
            bucket_stats = {'hard': {'count': 0, 'scores': []}, 'easy': {'count': 0, 'scores': []}}
            try:
                with open(manifest_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        bucket = row.get('bucket', 'unknown')
                        if bucket in bucket_stats:
                            bucket_stats[bucket]['count'] += 1
                            # Compute mosquito score if we can
                            wav_path = Path("data/processed") / row['path']
                            if wav_path.exists():
                                try:
                                    audio, sr = sf.read(str(wav_path), dtype='float32', always_2d=False)
                                    # Ensure mono
                                    if audio.ndim == 2:
                                        audio = audio.mean(axis=1)
                                    # Skip files that don't match expected format
                                    if sr != 16000 or len(audio) != 16000:
                                        continue
                                    score = mosquito_score(audio, sr)
                                    bucket_stats[bucket]['scores'].append(score)
                                except Exception as e:
                                    pass
                
                # Report bucket counts
                total = sum(b['count'] for b in bucket_stats.values())
                print(f"\nBucket Distribution:")
                for bucket in ['hard', 'easy']:
                    count = bucket_stats[bucket]['count']
                    pct = (count / total * 100) if total > 0 else 0
                    print(f"  {bucket.upper()}: {count} files ({pct:.1f}%)")
                
                # Report mosquito score by bucket
                print(f"\nMosquito Score by Bucket:")
                for bucket in ['hard', 'easy']:
                    scores = bucket_stats[bucket]['scores']
                    if scores:
                        scores_arr = np.array(scores)
                        high_count = (scores_arr > 0.12).sum()
                        high_pct = (high_count / len(scores) * 100) if len(scores) > 0 else 0
                        print(f"  {bucket.upper()}:")
                        print(f"    Mean:   {scores_arr.mean():.4f}")
                        print(f"    Median: {np.median(scores_arr):.4f}")
                        print(f"    P25:    {np.percentile(scores_arr, 25):.4f}")
                        print(f"    P75:    {np.percentile(scores_arr, 75):.4f}")
                        print(f"    Min:    {scores_arr.min():.4f}")
                        print(f"    Max:    {scores_arr.max():.4f}")
                        print(f"    High mosquito-like (>0.12): {high_count}/{len(scores)} ({high_pct:.1f}%)")
                    else:
                        print(f"  {bucket.upper()}: No scores computed")
                        
            except Exception as e:
                print(f"Error reading manifest: {e}")
        else:
            print(f"Manifest not found: {manifest_path}")
            print("Cannot compute bucket statistics without manifest.")
    # Audio content preview
    print("\n" + "="*80)
    print("CHECK 3: Audio content preview")
    print("="*80)
    print(f"\nAnalyzing {SAMPLES_PER_CLASS} random samples from each class:")
    print("(To enable actual playback: pip install sounddevice and uncomment playback code)\n")
    for class_name, count in sorted(class_counts.items()):
        if DATASET.lower() == "background":
            class_dir = processed_dir
            wav_files = list(processed_dir.glob('*.wav'))
        else:
            class_dir = processed_dir / class_name
            wav_files = list(class_dir.glob('*.wav'))
        if len(wav_files) == 0:
            continue
        num_samples = min(SAMPLES_PER_CLASS, len(wav_files))
        sampled_files = random.sample(wav_files, num_samples)
        print(f"{class_name}:")
        for filepath in sampled_files:
            play_audio_sample(filepath)
    # Summary
    print("\n" + "="*80)
    print(" "*33 + "SANITY CHECK SUMMARY")
    print("="*80)
    if len(all_issues) == 0:
        print("\nAll checks passed! Dataset is ready for training.\n")
        print("Summary:")
        print(f"  - {len(class_counts)} classes found")
        print(f"  - {total_files} total windows")
        print(f"  - All sampled files have correct format (16kHz, 1.0s)")
        print(f"  - All sampled files have audible content")
    else:
        print(f"\nFound {len(all_issues)} issues:\n")
        for issue in all_issues[:20]:
            print(f"  - {issue}")
        if len(all_issues) > 20:
            print(f"  ... and {len(all_issues) - 20} more issues")
        print("\nPlease review the issues above before training.")
    print("="*80 + "\n")
    # Recommendations for common issues
    print("RECOMMENDATIONS:")
    print("  1. If classes are very imbalanced, consider:")
    print("     - Using class weights during training")
    print("     - Data augmentation for minority classes")
    print("  2. If you find silent windows:")
    print("     - Increase RMS_THRESHOLD in preprocess_wingbeats.py")
    print("  3. Listen to a few samples manually to verify buzz quality")
    print("  4. Check average windows per recording:")
    if len(class_counts) > 0:
        for class_name in sorted(class_counts.keys()):
            if DATASET.lower() == "background":
                class_dir = processed_dir
                wav_files = list(processed_dir.glob('*.wav'))
            else:
                class_dir = processed_dir / class_name
                wav_files = list(class_dir.glob('*.wav'))
            recordings = set()
            for f in wav_files:
                # Match filename patterns: __clip for Wingsbeats, __idx for Abuzz/Background
                if '__clip' in f.stem:
                    parts = f.stem.rsplit('__clip', 1)
                elif '__idx' in f.stem:
                    parts = f.stem.rsplit('__idx', 1)
                else:
                    parts = [f.stem]  # Use full stem if no known pattern
                
                if len(parts) >= 1:
                    recordings.add(parts[0])
            if len(recordings) > 0:
                avg_windows = len(wav_files) / len(recordings)
                print(f"     {class_name}: ~{avg_windows:.1f} windows per recording ({len(recordings)} recordings)")
    print("\n")

def main():
    random.seed(42)
    sanity_check()

if __name__ == "__main__":
    main()
