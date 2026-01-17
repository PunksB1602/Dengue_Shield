from pathlib import Path

def touch(path: Path, content: str = "", overwrite: bool = False) -> None:
    """Create file with content if it doesn't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not path.exists():
        path.write_text(content, encoding="utf-8")

def create_structure():
    """
    Creates project structure for Dengue-Shield.
    Safe .gitignore, centralized config, correct audio params.
    """

    root = Path(".")

    # 1. Directory tree
    structure = [
        # Data Folders
        "data/raw/wingbeats",
        "data/raw/abuzz",
        "data/interim/abuzz_segments",
        "data/interim/mels_cache",
        "data/processed",
        "data/splits",
        "data/external",

        # Source Code
        "src",
        "src/data",
        "src/models",
        "src/train",
        "src/infer",
        "src/utils",

        # Experiments & Artifacts
        "saved_models",
        "runs",
        "logs",

        # Demo / App
        "app",
        "app/assets",
        "assets",
        "assets/demo_audio",

        # Docs / Scripts
        "docs",
        "docs/papers",
        "reports",
        "notebooks",
        "scripts",

        # Configs
        "configs",
        "tests",
    ]

    for folder in structure:
        p = root / folder
        created = not p.exists()
        p.mkdir(parents=True, exist_ok=True)
        if created:
            print(f"Created: {folder}/")

    # 2. .gitkeep for empty folders
    gitkeep_dirs = [
        "data/raw/wingbeats",
        "data/raw/abuzz",
        "data/interim/abuzz_segments",
        "data/interim/mels_cache",
        "data/processed",
        "data/external",
        "data/splits",
        "saved_models",
        "runs",
        "logs",
        "app/assets",
        "assets",
        "assets/demo_audio",
        "reports",
    ]
    for d in gitkeep_dirs:
        touch(root / d / ".gitkeep")

    # 3. __init__.py files
    init_files = [
        "src/__init__.py",
        "src/data/__init__.py",
        "src/models/__init__.py",
        "src/train/__init__.py",
        "src/infer/__init__.py",
        "src/utils/__init__.py",
    ]
    for f in init_files:
        p = root / f
        if not p.exists():
            touch(p)
            print(f"Created: {f}")

    # 4. .gitignore
    gitignore_content = """\
# Python / Environments
__pycache__/
*.pyc
.venv/
venv/
.env
.ipynb_checkpoints/
.pytest_cache/

# Data - ignore heavy stuff, keep structure
data/raw/**
data/processed/**
data/interim/**
data/external/**

# But track splits and manifests
!data/splits/
!data/splits/**
!data/**/manifest*.csv
!data/**/manifest*.json
!**/.gitkeep

# Models & Artifacts
saved_models/
runs/
logs/
*.pth
*.pt
*.onnx

# Archives
*.zip
*.rar
*.7z
*.tar
*.tar.gz
*.tgz

# Audio files (only under data/)
data/**/*.wav
data/**/*.flac
data/**/*.mp3
data/**/*.m4a
data/**/*.aac
data/**/*.ogg

# Allow demo audio in assets
!assets/
!assets/**
!assets/**/*.wav
!assets/**/*.mp3
!assets/**/*.flac
!assets/**/*.m4a
!assets/**/*.aac
!assets/**/*.ogg
!app/assets/
!app/assets/**
!app/assets/**/*.wav
!app/assets/**/*.mp3
!app/assets/**/*.flac
!app/assets/**/*.m4a
!app/assets/**/*.aac
!app/assets/**/*.ogg

# OS / Editor
.DS_Store
Thumbs.db
.vscode/
.idea/
"""
    path = root / ".gitignore"
    msg = "Created" if not path.exists() else "Updated"
    path.write_text(gitignore_content, encoding="utf-8")
    print(f"{msg}: .gitignore")

    # 5. requirements.txt
    requirements = """\
torch
torchvision
torchaudio
librosa>=0.10,<0.11
numpy>=1.23,<2.0
pandas
scikit-learn
scipy
soundfile
sounddevice>=0.4,<0.5
streamlit>=1.28,<2.0
tqdm
pyyaml
matplotlib
protobuf<5
"""
    path = root / "requirements.txt"
    msg = "Created" if not path.exists() else "Updated"
    path.write_text(requirements, encoding="utf-8")
    print(f"{msg}: requirements.txt")

    # 6. Config
    default_config = """\
# Audio params (must match between train & inference)
audio:
  sample_rate: 16000
  stream_window_sec: 1.0   # Streaming: 1s chunks
  stream_hop_sec: 0.5      # Streaming: 50% overlap
  n_mels: 128
  n_fft: 1024
  stft_win_length: 400     # STFT: 25ms window
  stft_hop_length: 160     # STFT: 10ms hop
  fmax: 4000               # 4kHz captures mosquito harmonics
  center: false            # No padding for streaming consistency

# MODEL CONFIG
model:
  name: resnet18
  num_classes: 4           # 3 species + 1 background

# TRAINING
training:
  batch_size: 32
  epochs: 50
  lr: 0.001
  device: auto             # auto/cuda/cpu

# LABELS
labels:
  map:
    Aedes: 0
    Anopheles: 1
    Culex: 2
    Background: 3
  names: ["Aedes", "Anopheles", "Culex", "Background"]

# Live inference settings
inference:
  risk_buffer_sec: 10
  mean_thresh: 0.6
  max_thresh: 0.8
  cooldown_sec: 10

# PATHS
paths:
  wingbeats_root: data/raw/wingbeats
  abuzz_root: data/raw/abuzz
  splits_dir: data/splits
  models_dir: saved_models

# REPRODUCIBILITY
seed: 42
"""
    path = root / "configs" / "default.yaml"
    msg = "Created" if not path.exists() else "Updated"
    path.write_text(default_config, encoding="utf-8")
    print(f"{msg}: configs/default.yaml")

    # 7. Script stubs
    touch(root / "scripts" / "make_splits_wingbeats.py", "# TODO: create train/val split CSVs\n")
    touch(root / "src" / "train" / "train_main.py", "# TODO: training entrypoint\n")
    touch(root / "app" / "app.py", "# TODO: streamlit demo entrypoint\n")
    
    # Config validator
    config_validator = '''
import yaml
from pathlib import Path

def load_config(path="configs/default.yaml"):
    """Load and validate config."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    
    # Check required sections
    required = ["audio", "model", "labels", "training"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Config missing sections: {missing}")
    
    # Validate audio params
    audio = cfg["audio"]
    if audio.get("sample_rate", 0) <= 0:
        raise ValueError("audio.sample_rate must be > 0")
    if audio.get("fmax", 0) <= 0:
        raise ValueError("audio.fmax must be > 0")
    
    # Validate model
    if cfg["model"].get("num_classes", 0) <= 0:
        raise ValueError("model.num_classes must be > 0")
    
    # Validate labels
    labels = cfg["labels"]
    if len(labels.get("names", [])) != cfg["model"]["num_classes"]:
        raise ValueError(f"labels.names length must match model.num_classes")
    
    return cfg
'''
    touch(root / "src" / "utils" / "config.py", config_validator, overwrite=True)
    
    # Audio utility functions
    audio_utils = '''
import librosa
import numpy as np
from pathlib import Path

def load_audio(path, sr=16000, duration=1.0):
    """Load audio file and return fixed-length segment."""
    # TODO: implement with librosa.load
    # Should return exactly sr * duration samples (pad/trim)
    pass

def melspec(audio, sr=16000, n_mels=128, n_fft=1024, 
            hop_length=160, win_length=400, fmax=4000, center=False):
    """Convert audio to mel spectrogram.
    
    Note: use stft_* params from config, not stream_* params.
    Always pass center=False (librosa defaults to True).
    """ 
    # TODO: implement with librosa.feature.melspectrogram
    pass

def normalize(spec, method='db'):
    """Normalize spectrogram.""" 
    # TODO: implement power_to_db or min-max normalization
    pass
'''
    touch(root / "src" / "utils" / "audio.py", audio_utils, overwrite=True)
    
    # Streaming inference buffer
    streaming_stub = '''
import numpy as np
from collections import deque

class RollingBuffer:
    """Circular buffer for streaming audio and predictions."""
    def __init__(self, window_sec=10, hop_sec=0.5, sr=16000):
        self.audio_buffer = deque(maxlen=int(window_sec / hop_sec))
        self.pred_buffer = deque(maxlen=int(window_sec / hop_sec))
        self.sr = sr
    
    def push_audio(self, audio_chunk):
        """Add new audio chunk.""" 
        # TODO: implement
        pass
    
    def push_pred(self, probs):
        """Store prediction for this chunk.""" 
        # TODO: implement
        pass
    
    def recent_preds(self):
        """Get recent predictions for temporal aggregation.""" 
        # TODO: implement
        pass
'''
    touch(root / "src" / "infer" / "streaming.py", streaming_stub, overwrite=True)
    
    # .env.example
    env_example = '''
# Device settings (optional)
CUDA_VISIBLE_DEVICES=0

# Custom paths (optional)
# WINGBEATS_ROOT=data/raw/wingbeats
# ABUZZ_ROOT=data/raw/abuzz
'''
    touch(root / ".env.example", env_example, overwrite=True)
    
    # Cross-platform clean script
    clean_script = '''
from pathlib import Path
import shutil

print("Cleaning Python artifacts...")

# Remove __pycache__ directories
for p in Path(".").rglob("__pycache__"):
    shutil.rmtree(p, ignore_errors=True)
    print(f"Removed: {p}")

# Remove .pyc files
for p in Path(".").rglob("*.pyc"):
    p.unlink(missing_ok=True)

print("Done!")
'''
    touch(root / "scripts" / "clean.py", clean_script, overwrite=True)
    
    # Makefile (Windows-compatible clean)
    makefile = '''
.PHONY: setup splits train demo clean

setup:
	pip install -r requirements.txt

splits:
	python scripts/make_splits_wingbeats.py

train:
	PYTHONPATH=. python -m src.train.train_main

demo:
	streamlit run app/app.py

clean:
	python scripts/clean.py
'''
    touch(root / "Makefile", makefile, overwrite=True)

    # 8. README
    readme = """\
# Dengue-Shield

## Data Setup
1. **Wingbeats**: Extract to `data/raw/wingbeats/`
2. **ABUZZ**: Extract to `data/raw/abuzz/`
3. **Important**: Do not `git add data/` (raw/interim/processed are ignored by design)

## Key Commands
```bash
# Install dependencies
make setup
# or: pip install -r requirements.txt

# Create train/val splits
python scripts/make_splits_wingsbeats.py

# Train model (run from repo root)
PYTHONPATH=. python -m src.train.train_main
# or: make train

# Launch demo
make demo
# or: streamlit run app/app.py
```

## Config
Edit `configs/default.yaml` to change audio parameters.
"""
    path = root / "README.md"
    msg = "Created" if not path.exists() else "Updated"
    path.write_text(readme, encoding="utf-8")
    print(f"{msg}: README.md")

    print("\nPROJECT STRUCTURE & CONFIGS READY")
    print("---------------------------------------------------")
    print("1. Run: pip install -r requirements.txt")
    print("2. Move your datasets into data/raw/")
    print("3. You are ready to start coding.")

if __name__ == "__main__":
    create_structure()


