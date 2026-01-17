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
import numpy as np
import soundfile as sf
from collections import defaultdict

# Constants
TARGET_SR = 16000
TARGET_DURATION = 1.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)
SAMPLES_PER_CLASS = 10

# Mosquito-band parameters (consistent with preprocess scripts)
FMAX = 4000
BAND_LO, BAND_HI = 250, 1200  # Match ABUZZ/Wingsbeats preprocess
MOSQUITO_SCORE_THRESHOLD_MIN = 0.02  # Min for mosquito datasets
MOSQUITO_SCORE_THRESHOLD_MAX = 0.12  # Max for background dataset


def count_files_per_class(processed_dir):
    """Count number of WAV files in each class directory."""
    class_counts = {}
    
    for class_dir in sorted(processed_dir.iterdir()):
        if class_dir.is_dir():
            wav_files = list(class_dir.glob('*.wav'))
            class_counts[class_dir.name] = len(wav_files)
    
    return class_counts


def mosquito_score(window, sr=TARGET_SR):
    """Ratio of energy in 250-1200 Hz to total 0-4kHz."""
    x = window - np.mean(window)
    spec = np.abs(np.fft.rfft(x))**2
    freqs = np.fft.rfftfreq(len(x), d=1.0/sr)
    mask_all = (freqs >= 0) & (freqs <= FMAX)
    mask_band = (freqs >= BAND_LO) & (freqs <= BAND_HI)
    return float(spec[mask_band].sum() / (spec[mask_all].sum() + 1e-12))


def check_file_duration_and_content(filepath, check_mode='rms_only'):
    """Check if file has correct format and audible content.
    
    Args:
        filepath: Path to audio file
        check_mode: 'mosquito_positive', 'mosquito_negative', or 'rms_only'
    
    Returns:
        tuple: (format_ok, has_content, sr, num_samples, rms, peak, m_score)
    """
    try:
        audio, sr = sf.read(filepath, dtype='float32')
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        sr_ok = (sr == TARGET_SR)
        samples_ok = (len(audio) == TARGET_SAMPLES)
        format_ok = sr_ok and samples_ok
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.abs(audio).max()
        
        m_score = None
        if check_mode == 'mosquito_positive':
            m_score = mosquito_score(audio, sr=sr)
            has_content = (rms > 0.001) and (m_score > MOSQUITO_SCORE_THRESHOLD_MIN)
        elif check_mode == 'mosquito_negative':
            m_score = mosquito_score(audio, sr=sr)
            has_content = (rms > 0.001) and (m_score < MOSQUITO_SCORE_THRESHOLD_MAX)
        else:  # rms_only
            has_content = rms > 0.001
        
        return format_ok, has_content, sr, len(audio), rms, peak, m_score
    except Exception:
        return False, False, 0, 0, 0.0, 0.0, None


def play_audio_sample(filepath):
    """
    Simulate playing audio (prints info instead).
    On Windows, you could use winsound or pygame for actual playback.
    """
    try:
        audio, sr = sf.read(filepath, dtype='float32')
        
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        
        duration = len(audio) / sr
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.abs(audio).max()
        
        print(f"    {filepath.name}")
        print(f"      Duration: {duration:.3f}s | RMS: {rms:.4f} | Peak: {peak:.4f}")
        
        # Optional: Uncomment to actually play on Windows
        # import sounddevice as sd
        # sd.play(audio, sr)
        # sd.wait()
        
    except Exception as e:
        print(f"    ERROR: {filepath.name} - {str(e)}")


def sanity_check():
    """Run comprehensive sanity checks on processed dataset."""
    DATASET = sys.argv[1] if len(sys.argv) > 1 else "Abuzz"
    
    # Robust path resolver (case-insensitive)
    candidates = [
        Path(f"data/processed/{DATASET}"),
        Path(f"data/processed/{DATASET.lower()}"),
        Path(f"data/processed/{DATASET.capitalize()}"),
    ]
    
    processed_dir = None
    if DATASET.lower() == "background":
        processed_dir = Path("data/processed/Background/Background")
    else:
        for p in candidates:
            if p.exists():
                processed_dir = p
                break
        if processed_dir is None:
            processed_dir = candidates[0]  # fallback
    
    # Determine check mode based on dataset
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
        print(f"Using mosquito-band positive check (score > {MOSQUITO_SCORE_THRESHOLD_MIN})\n")
    elif check_mode == 'mosquito_negative':
        print(f"Using mosquito-band negative check (score < {MOSQUITO_SCORE_THRESHOLD_MAX})\n")
    else:
        print(f"Using RMS content check only\n")
    
    # Check 1: Count files per class
    print("="*80)
    print("CHECK 1: File counts per class")
    print("="*80)
    
    # Special handling for Background (no class subdirectories)
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
    
    # Check for balance
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
    
    # Check 2: Sample random files and verify content
    print("\n" + "="*80)
    print("CHECK 2: Random sample verification")
    print("="*80)
    
    all_issues = []
    class_stats = {}  # Track stats per class
    
    for class_name, count in sorted(class_counts.items()):
        # Get wav files for this class
        if DATASET.lower() == "background":
            class_dir = processed_dir
            wav_files = list(processed_dir.glob('*.wav'))
        else:
            class_dir = processed_dir / class_name
            wav_files = list(class_dir.glob('*.wav'))
        
        if len(wav_files) == 0:
            print(f"\n{class_name}: No files found!")
            continue
        
        # Sample random files
        num_samples = min(SAMPLES_PER_CLASS, len(wav_files))
        sampled_files = random.sample(wav_files, num_samples)
        
        print(f"\n{class_name}: Checking {num_samples} random samples")
        
        format_issues = 0
        content_issues = 0
        peak_values = []
        rms_values = []
        score_values = []
        
        for filepath in sampled_files:
            format_ok, has_content, sr, num_samples_file, rms, peak, m_score = check_file_duration_and_content(filepath, check_mode)
            
            peak_values.append(peak)
            rms_values.append(rms)
            if m_score is not None:
                score_values.append(m_score)
            
            if not format_ok:
                format_issues += 1
                all_issues.append(f"{class_name}/{filepath.name}: Wrong format (SR={sr}, samples={num_samples_file})")
            
            if not has_content:
                content_issues += 1
                if m_score is not None:
                    all_issues.append(f"{class_name}/{filepath.name}: low content (rms={rms:.6f}, mscore={m_score:.4f})")
                else:
                    all_issues.append(f"{class_name}/{filepath.name}: low RMS (rms={rms:.6f})")
        
        # Store stats for this class
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
        
        # Check for excessive normalization (clipping)
        if len(peak_values) > 0:
            avg_peak = np.mean(peak_values)
            high_peak_count = sum(1 for p in peak_values if p > 0.9)
            
            if high_peak_count > num_samples * 0.8:  # More than 80% near clipping
                print(f" {high_peak_count}/{num_samples} files have peak > 0.9 (avg peak: {avg_peak:.3f})")
                print(f" This suggests aggressive normalization. Check if this is expected.")
            else:
                print(f" Peak levels reasonable (avg: {avg_peak:.3f})")
    
    # Quick stats report
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
    
    # Check 3: Audio content preview
    print("\n" + "="*80)
    print("CHECK 3: Audio content preview")
    print("="*80)
    print(f"\nAnalyzing {SAMPLES_PER_CLASS} random samples from each class:")
    print("(To enable actual playback: pip install sounddevice and uncomment playback code)\n")
    
    for class_name, count in sorted(class_counts.items()):
        # Get wav files for this class
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
        for issue in all_issues[:20]:  # Show first 20
            print(f"  - {issue}")
        if len(all_issues) > 20:
            print(f"  ... and {len(all_issues) - 20} more issues")
        print("\nPlease review the issues above before training.")
    
    print("="*80 + "\n")
    
    # Additional recommendations
    print("RECOMMENDATIONS:")
    print("  1. If classes are very imbalanced, consider:")
    print("     - Using class weights during training")
    print("     - Data augmentation for minority classes")
    print("  2. If you find silent windows:")
    print("     - Increase RMS_THRESHOLD in preprocess_wingbeats.py")
    print("  3. Listen to a few samples manually to verify buzz quality")
    print("  4. Check average windows per recording:")
    
    if len(class_counts) > 0:
        # Estimate recordings based on filename patterns
        for class_name in sorted(class_counts.keys()):
            # Special handling for Background (no class subdirectories)
            if DATASET.lower() == "background":
                class_dir = processed_dir
                wav_files = list(processed_dir.glob('*.wav'))
            else:
                class_dir = processed_dir / class_name
                wav_files = list(class_dir.glob('*.wav'))
            
            # Extract unique recording names (everything before __idx/__seg/__win)
            recordings = set()
            for f in wav_files:
                parts = f.stem.rsplit('__idx', 1)
                if len(parts) == 1:
                    parts = f.stem.rsplit('__seg', 1)  # backward compatibility
                if len(parts) == 1:
                    parts = f.stem.rsplit('__win', 1)  # backward compatibility
                if len(parts) == 2:
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


