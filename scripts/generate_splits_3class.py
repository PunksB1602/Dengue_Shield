"""
Generate 3-Class (Aedes, Non-Aedes, Background) splits for Pre-training and Fine-tuning.
Split Ratio: 70% Train / 15% Val / 15% Test
Usage:
    python scripts/generate_splits_3class.py
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# --- CONFIG ---
SPLITS_DIR = Path('data/splits')
OUTPUT_DIR = Path('data/splits_3class')
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42

# 3-Class Mapping
# Aedes -> 0
# Non-Aedes (Anopheles, Culex) -> 1
# Background -> 2
def map_label(original_label):
    if original_label == 'Aedes':
        return 0
    elif original_label in ['Anopheles', 'Culex']:
        return 1
    elif original_label == 'Background':
        return 2
    else:
        return 2 # Fallback

def split_dataframe(df, name):
    """Splits DF into 70/15/15."""
    # First, 70% Train, 30% Temp
    train, temp = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED, shuffle=True)
    # Then split Temp into 50/50 (15% Val, 15% Test)
    val, test = train_test_split(temp, test_size=0.5, random_state=RANDOM_SEED, shuffle=True)
    
    print(f"  {name}: {len(df)} total -> {len(train)} Train, {len(val)} Val, {len(test)} Test")
    return train, val, test

def main():
    print("="*60)
    print(" GENERATING 3-CLASS SPLITS (70/15/15)")
    print("="*60)

    # 1. Load Manifests
    try:
        wb_df = pd.read_csv(SPLITS_DIR / 'wingsbeats_manifest.csv')
        ab_df = pd.read_csv(SPLITS_DIR / 'abuzz_manifest.csv')
        bg_df = pd.read_csv(SPLITS_DIR / 'background_manifest.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the preprocessing scripts first!")
        return

    # 2. Apply Class Mapping
    wb_df['label_idx'] = wb_df['label'].apply(map_label)
    ab_df['label_idx'] = ab_df['label'].apply(map_label)
    bg_df['label_idx'] = bg_df['label'].apply(map_label)

    # 3. Perform Splits (70/15/15)
    print("\nSplitting Datasets:")
    wb_train, wb_val, wb_test = split_dataframe(wb_df, "Wingbeats")
    bg_train, bg_val, bg_test = split_dataframe(bg_df, "Background")
    ab_train, ab_val, ab_test = split_dataframe(ab_df, "Abuzz")

    # 4. Construct Pre-training Sets (Wingbeats + Background)
    # We use the BACKGROUND splits here.
    pre_train = pd.concat([wb_train, bg_train]).sample(frac=1, random_state=RANDOM_SEED)
    pre_val   = pd.concat([wb_val, bg_val]).sample(frac=1, random_state=RANDOM_SEED)
    pre_test  = pd.concat([wb_test, bg_test]).sample(frac=1, random_state=RANDOM_SEED)

    # 5. Construct Fine-tuning Sets (Abuzz + Background)
    # CRITICAL: Reuse the SAME Background splits to avoid data leakage.
    # We are "teaching" the model new mosquito domains (Abuzz) while "reminding" it of background.
    ft_train = pd.concat([ab_train, bg_train]).sample(frac=1, random_state=RANDOM_SEED)
    ft_val   = pd.concat([ab_val, bg_val]).sample(frac=1, random_state=RANDOM_SEED)
    ft_test  = pd.concat([ab_test, bg_test]).sample(frac=1, random_state=RANDOM_SEED)

    # 6. Save Files
    cols = ['path', 'label_idx', 'species', 'source']
    
    # Pre-train Files
    pre_train[cols].to_csv(OUTPUT_DIR / 'pretrain_train.csv', index=False)
    pre_val[cols].to_csv(OUTPUT_DIR / 'pretrain_val.csv', index=False)
    pre_test[cols].to_csv(OUTPUT_DIR / 'pretrain_test.csv', index=False)

    # Fine-tune Files
    ft_train[cols].to_csv(OUTPUT_DIR / 'finetune_train.csv', index=False)
    ft_val[cols].to_csv(OUTPUT_DIR / 'finetune_val.csv', index=False)
    ft_test[cols].to_csv(OUTPUT_DIR / 'finetune_test.csv', index=False)

    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    print(f"Files saved to: {OUTPUT_DIR}")
    print("\nPRE-TRAIN (Wingbeats + BG):")
    print(f"  Train: {len(pre_train)} | Val: {len(pre_val)} | Test: {len(pre_test)}")
    print("\nFINE-TUNE (Abuzz + BG):")
    print(f"  Train: {len(ft_train)} | Val: {len(ft_val)} | Test: {len(ft_test)}")
    print(f"\nClass Mapping: 0=Aedes, 1=Non-Aedes, 2=Background")

if __name__ == "__main__":
    main()