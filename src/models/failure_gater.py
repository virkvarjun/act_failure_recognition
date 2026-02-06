import torch
import torch.nn as nn
import numpy as np
from collections import deque
from pathlib import Path

class FailureClassifier(nn.Module):
    """Mirror the architecture from training."""
    def __init__(self, input_dim=96, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class FailureGater:
    def __init__(self, model_path, threshold=0.29, window_size=8, n_features=12, hysteresis_count=3):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = FailureClassifier(input_dim=window_size * n_features)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.threshold = threshold
        self.window_size = window_size
        self.n_features = n_features
        self.hysteresis_count = hysteresis_count
        
        self.window = deque(maxlen=window_size)
        self.alarm_counter = 0
        self.is_unsafe = False

    def update((self, state, action)):
        """
        state: np.array (6,)
        action: np.array (6,)
        Returns: bool (True if UNSAFE triggered)
        """
        # Feature vector for current step
        features = np.concatenate([state, action]) # (12,)
        self.window.append(features)
        
        if len(self.window) < self.window_size:
            return False
            
        # Prepare input
        input_tensor = torch.FloatTensor(np.array(self.window)).reshape(1, -1).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            prob_unsafe = torch.softmax(outputs, dim=1)[0, 1].item()
            
        # Hysteresis logic
        if prob_unsafe >= self.threshold:
            self.alarm_counter += 1
        else:
            self.alarm_counter = 0 # Immediate reset if one safe step found? Or gradual? Let's go immediate for safety.
            
        if self.alarm_counter >= self.hysteresis_count:
            self.is_unsafe = True
        else:
            self.is_unsafe = False
            
        return self.is_unsafe

    def reset(self):
        self.window.clear()
        self.alarm_counter = 0
        self.is_unsafe = False
