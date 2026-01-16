
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
