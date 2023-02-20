from dataclasses import dataclass

import gym
import mujoco_py
import torch
from datasets import load_dataset
from transformers import (
    DecisionTransformerself,
)
from colabgymrender.recorder import Recorder


class TrainableDT(DecisionTransformerself):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        # add the DT loss
        action_preds = output[1]
        action_targets = kwargs["actions"]
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[
            attention_mask.reshape(-1) > 0
        ]

        loss = torch.mean((action_preds - action_targets) ** 2)

        return {"loss": loss}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)

    # function that gets an action from the model using autoregressive prediction with a window of the previous 20 timesteps.
    def get_action(self, states, actions, rewards, returns_to_go, timesteps):
        # This implementation does not condition on past rewards

        states = states.reshape(1, -1, self.config.state_dim)
        actions = actions.reshape(1, -1, self.config.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        states = states[:, -self.config.max_length :]
        actions = actions[:, -self.config.max_length :]
        returns_to_go = returns_to_go[:, -self.config.max_length :]
        timesteps = timesteps[:, -self.config.max_length :]

        # pad all tokens to sequence length
        padding = self.config.max_length - states.shape[1]
        attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
        attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)

        states = torch.cat(
            [torch.zeros((1, padding, self.config.state_dim)), states], dim=1
        ).float()
        actions = torch.cat(
            [torch.zeros((1, padding, self.config.act_dim)), actions], dim=1
        ).float()
        returns_to_go = torch.cat(
            [torch.zeros((1, padding, 1)), returns_to_go], dim=1
        ).float()
        timesteps = torch.cat(
            [torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1
        )

        state_preds, action_preds, return_preds = self.original_forward(
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,
        )

        return action_preds[0, -1]
