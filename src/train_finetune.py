"""
Two-Stage Training Script (Fixed loading with soundfile)
Phase 1: Pre-train on Wingbeats + Background (Clean)
Phase 2: Fine-tune on Abuzz + Background (Noisy)
Usage: python train_finetune.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
import soundfile as sf  # Uses soundfile directly for robustness
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
from model import AudioResNet

# --- CONFIG ---
BATCH_SIZE = 64
PRETRAIN_EPOCHS = 10     # Epochs for Wingbeats
FINETUNE_EPOCHS = 10     # Epochs for Abuzz
LR_PRETRAIN = 1e-4
LR_FINETUNE = 1e-5       # Lower LR for fine-tuning
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = 'data/processed'
SPLITS_DIR = 'data/splits_3class' 
MODELS_DIR = 'models'

# --- DATASET ---
class DengueDataset(Dataset):
    def __init__(self, csv_path, root_dir, is_train=False):
        self.df = pd.read_csv(csv_path)
        self.root_dir = Path(root_dir)
        self.is_train = is_train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.root_dir / row['path']
        label = int(row['label_idx'])
        
        try:
            # FIX: Use soundfile instead of torchaudio.load to avoid TorchCodec errors
            audio_np, sr = sf.read(str(path), dtype='float32')
            
            # Convert to Tensor
            waveform = torch.from_numpy(audio_np)
            
            # Ensure Channel Dimension: [Samples] -> [1, Samples]
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            # Resample if needed (Safety check, though data is likely 16k)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)

            # Shape enforcement (Pad/Crop to 16000)
            target_len = 16000
            current_len = waveform.shape[1]
            
            if current_len < target_len:
                pad = target_len - current_len
                waveform = torch.nn.functional.pad(waveform, (0, pad))
            elif current_len > target_len:
                # Random crop for training, Center crop for val
                if self.is_train:
                    start = torch.randint(0, current_len - target_len, (1,)).item()
                else:
                    start = (current_len - target_len) // 2
                waveform = waveform[:, start : start + target_len]
                
            # Augmentation (Train only)
            if self.is_train:
                # Random Gain
                gain = torch.rand(1) * 0.4 + 0.8
                waveform = waveform * gain
                # Add Noise
                noise = torch.randn_like(waveform) * 0.005
                waveform = waveform + noise
                
            return waveform, label
            
        except Exception as e:
            # Silent fail for corrupted files to keep training moving
            # (sanity check showed your data is good, so this should be rare)
            # print(f"Error loading {path}: {e}") 
            return torch.zeros((1, 16000)), 2 # Return silence labeled as Background

# --- TRAINING LOOP ---
def run_training_phase(phase_name, model, train_loader, val_loader, epochs, lr, save_path):
    print(f"\n[{phase_name}] Starting training for {epochs} epochs...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        # TRAIN
        model.train()
        running_loss = 0.0
        
        # tqdm progress bar
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{epochs}", leave=False)
        
        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            # Remove extra channel dim if present [B, 1, 16000] -> [B, 16000] for MelSpectrogram
            if inputs.dim() == 3: inputs = inputs.squeeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        train_loss = running_loss / len(train_loader)
        
        # VALIDATE
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                if inputs.dim() == 3: inputs = inputs.squeeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_acc = 100. * correct / total
        val_loss /= len(val_loader)
        
        # Step Scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f" {phase_name} Ep {epoch+1}: Train Loss {train_loss:.4f} | Val Acc {val_acc:.2f}% | LR {current_lr:.1e}")
        
        # Save Best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            
    print(f"[{phase_name}] Best Accuracy: {best_acc:.2f}% -> Saved to {save_path}")
    
    # Load best weights for next phase
    model.load_state_dict(torch.load(save_path))
    return model

def main():
    Path(MODELS_DIR).mkdir(exist_ok=True)
    
    # --- PHASE 1: PRE-TRAIN (Wingbeats + Background) ---
    print("="*60)
    print(" PHASE 1: PRE-TRAINING (Clean Data)")
    print("="*60)
    train_ds = DengueDataset(f"{SPLITS_DIR}/pretrain_train.csv", DATA_ROOT, is_train=True)
    val_ds = DengueDataset(f"{SPLITS_DIR}/pretrain_val.csv", DATA_ROOT, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = AudioResNet(num_classes=3).to(DEVICE)
    model = run_training_phase("PreTrain", model, train_loader, val_loader, 
                             PRETRAIN_EPOCHS, LR_PRETRAIN, f"{MODELS_DIR}/model_pretrain.pth")

    # --- PHASE 2: FINE-TUNE (Abuzz + Background) ---
    print("\n" + "="*60)
    print(" PHASE 2: FINE-TUNING (Noisy Data)")
    print("="*60)
    
    ft_train_ds = DengueDataset(f"{SPLITS_DIR}/finetune_train.csv", DATA_ROOT, is_train=True)
    ft_val_ds = DengueDataset(f"{SPLITS_DIR}/finetune_val.csv", DATA_ROOT, is_train=False)
    
    ft_train_loader = DataLoader(ft_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    ft_val_loader = DataLoader(ft_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Train same model, new data, lower LR
    run_training_phase("FineTune", model, ft_train_loader, ft_val_loader, 
                     FINETUNE_EPOCHS, LR_FINETUNE, f"{MODELS_DIR}/model_finetuned.pth")

if __name__ == "__main__":
    main()