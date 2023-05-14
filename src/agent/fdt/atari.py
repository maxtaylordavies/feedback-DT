import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import FDTAgent


class AtariFDTAgent(FDTAgent):
    def create_state_embedding_model(self):
        # CNN for embedding image states
        self.state_embedding_model = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, self.hidden_size),
            nn.Tanh(),
        )
