from dataclasses import dataclass
from itertools import accumulate

import numpy as np
import torch

from src.utils import log, to_one_hot


@dataclass
class DecisionTransformerMinariDataCollator:
    def __init__(self, minari_dataset, max_sample_len=20, scale=1, gamma=1) -> None:
        self.max_sample_len, self.scale, self.gamma = max_sample_len, scale, gamma

        # compute start and end timesteps for each episode
        self.episode_ends = np.where(
            minari_dataset.terminations + minari_dataset.truncations == 1
        )[0]
        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1] + 1])
        self.num_episodes = len(self.episode_starts)

        self.success_indices = np.array(
            [i for i in range(self.num_episodes) if minari_dataset[i].rewards[-1] > 0]
        )
        log(f"Number of successful episodes: {len(self.success_indices)}")

        # # compute episode lengths, and thus define a distribution
        # # for sampling episodes with a probability proportional to their length
        # self.episode_lengths = self.episode_ends - self.episode_starts + 1
        # self.episode_probabilities = self.episode_lengths / sum(self.episode_lengths)

        # set state and action dimensions
        self.state_dim, self.act_dim = (
            np.prod(minari_dataset.get_observation_shape()),
            minari_dataset.get_action_size(),
        )

        # store observations, actions and rewards
        self.observations = minari_dataset.observations.reshape((-1, self.state_dim))
        self.actions = to_one_hot(
            minari_dataset.actions, self.act_dim
        )  # convert from index integer representation to one-hot
        self.rewards = minari_dataset.rewards

        # compute observation statistics
        self.state_mean, self.state_std = (
            np.mean(self.observations, axis=0),
            np.std(self.observations, axis=0) + 1e-6,  # avoid division by zero
        )

    def _normalise_states(self, states):
        return (states - self.state_mean) / self.state_std

    # helper func to pad 2D or 3D numpy array along axis 1
    def _pad(self, x, pad_width=None, before=True, val=0):
        pad_width = pad_width or max(self.max_sample_len - x.shape[1], 0)
        pad_shape = [(0, 0)] * len(x.shape)
        pad_shape[1] = (pad_width, 0) if before else (0, pad_width)
        return np.pad(x, pad_shape, constant_values=val)

    def _discounted_cumsum(self, x, gamma):
        return np.array(list(accumulate(x[::-1], lambda a, b: (gamma * a) + b)))[::-1]

    def _sample_batch(self, batch_size):
        t, s, a, r, rtg, mask = [], [], [], [], [], []

        # sample episodes with a probability proportional to their length
        # episode_indices = np.random.choice(np.arange(self.num_episodes), size=batch_size)
        episode_indices = np.random.choice(self.success_indices, size=batch_size)

        # sample a subsequence of each chosen episode
        for ep_idx in episode_indices:
            # sample an actual random start timestep for this episode
            # note: end represents the last timestep _included_ in the sequence,
            # which means when we're using exclusive range operators like [:]
            # or np.arange, we need to use end + 1
            start = np.random.randint(self.episode_starts[ep_idx], self.episode_ends[ep_idx])
            end = min(start + self.max_sample_len - 1, self.episode_ends[ep_idx])

            # store data
            t.append(self._pad(np.arange(0, end - start + 1).reshape(1, -1)))
            s.append(
                self._normalise_states(
                    self._pad(
                        self.observations[start : end + 1].reshape(1, -1, self.state_dim)
                    )
                )
            )
            a.append(
                self._pad(self.actions[start : end + 1].reshape(1, -1, self.act_dim), val=-10)
            )
            r.append(self._pad(self.rewards[start : end + 1].reshape(1, -1, 1)))
            rtg.append(
                self._pad(
                    self._discounted_cumsum(
                        self.rewards[start : self.episode_ends[ep_idx] + 1], gamma=self.gamma
                    )[: s[-1].shape[1]].reshape(1, -1, 1)
                )
                / self.scale
            )
            mask.append(
                np.concatenate(
                    [
                        np.zeros((1, self.max_sample_len - (end - start + 1))),
                        np.ones((1, end - start + 1)),
                    ],
                    axis=1,
                )
            )

        return {
            "timesteps": torch.from_numpy(np.concatenate(t, axis=0)).long(),
            "states": torch.from_numpy(np.concatenate(s, axis=0)).float(),
            "actions": torch.from_numpy(np.concatenate(a, axis=0)).float(),
            "rewards": torch.from_numpy(np.concatenate(r, axis=0)).float(),
            "returns_to_go": torch.from_numpy(np.concatenate(rtg, axis=0)).float(),
            "attention_mask": torch.from_numpy(np.concatenate(mask, axis=0)).float(),
        }

    def __call__(self, features):
        batch_size = len(features)
        return self._sample_batch(batch_size)
