"""
Evaluate the fine-tuned model on the Test Set.
Usage: python scripts/evaluate.py
"""
import torch
import pandas as pd
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from model import AudioResNet

# --- CONFIG ---
TEST_CSV = 'data/splits_3class/finetune_test.csv' # The noisy test set
MODEL_PATH = 'models/model_finetuned.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = 'data/processed'
CLASSES = ['Aedes', 'Non-Aedes', 'Background']

class TestDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = f"{DATA_ROOT}/{row['path']}"
        try:
            audio, sr = sf.read(path, dtype='float32')
            if len(audio) < 16000:
                audio = np.pad(audio, (0, 16000-len(audio)))
            elif len(audio) > 16000:
                audio = audio[:16000]
            return torch.from_numpy(audio).unsqueeze(0), int(row['label_idx'])
        except:
            return torch.zeros(1, 16000), 2

def main():
    print(f"Loading model from {MODEL_PATH}...")
    model = AudioResNet(num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    ds = TestDataset(TEST_CSV)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    all_preds = []
    all_labels = []

    print(f"Evaluating on {len(ds)} test samples...")
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Metrics
    print("\n" + "="*60)
    print(" CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\n" + "="*60)
    print(" CONFUSION MATRIX")
    print("="*60)
    print(f"{'':15} Predicted Aedes  Predicted Non-Aedes  Predicted BG")
    for i, row in enumerate(cm):
        print(f"Actual {CLASSES[i]:10} {row[0]:^15} {row[1]:^20} {row[2]:^12}")

if __name__ == "__main__":
    main()