
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
