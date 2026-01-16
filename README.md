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
python scripts/make_splits_wingbeats.py

# Train model (run from repo root)
PYTHONPATH=. python -m src.train.train_main
# or: make train

# Launch demo
make demo
# or: streamlit run app/app.py
```

## Config
Edit `configs/default.yaml` to change audio parameters.
