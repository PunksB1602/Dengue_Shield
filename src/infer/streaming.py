
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
