import torch.nn as nn
from torch.nn import functional as F

from .base import FDTAgent


class MinigridFDTAgent(FDTAgent):
    def create_state_embedding_model(self):
        # CNN for embedding image states
        self.state_embedding_model = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(40000, self.hidden_size),
            nn.Tanh(),
        )
