from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
from torch import nn


@dataclass
class AgentInput:
    mission_embeddings: Any
    states: Any
    actions: Any
    rewards: Any
    returns_to_go: Any
    timesteps: Any
    feedback_embeddings: Any
    attention_mask: Any


class Agent(nn.Module):
    """
    Agent is the base class used to represent any offline-rl-ish trainable agent.
    """

    def __init__(self) -> None:
        super().__init__()

    def _forward(self, input: AgentInput) -> Any:
        pass

    def _compute_loss(self, input: AgentInput, output: Any) -> float:
        pass

    def forward(self, input: AgentInput, **kwargs) -> Dict:
        output = self._forward(input)
        loss = self._compute_loss(input, output)
        return {"loss": loss}

    def get_action(self, input: AgentInput, context=1, one_hot=False):
        pass


class RandomAgent(Agent):
    def __init__(self, act_dim) -> None:
        self.act_dim = act_dim
        super().__init__()

    def _forward(self, input: AgentInput) -> Any:
        return None

    def _compute_loss(self, input: AgentInput, output: Any) -> float:
        return torch.tensor(0.0, requires_grad=True)

    def get_action(self, input: AgentInput, context=1, one_hot=False):
        return torch.tensor(
            np.random.random(self.act_dim).astype(np.float32),
            device=input.states.device,
        )
