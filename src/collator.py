from dataclasses import dataclass
from itertools import accumulate

import numpy as np
import torch

from src.utils import to_one_hot


@dataclass
class DecisionTransformerMinariDataCollator:
    def __init__(self, minari_dataset, max_sample_len=20, scale=1000.0, gamma=0.99) -> None:
        self.max_sample_len, self.scale, self.gamma = max_sample_len, scale, gamma
        self.num_episodes = len(minari_dataset)

        # compute start and end timesteps for each episode
        self.episode_ends = np.where(
            minari_dataset.terminations + minari_dataset.truncations == 1
        )[0]
        self.episode_starts = np.concatenate(
            [[0], self.episode_ends[: self.num_episodes - 1] + 1]
        )

        # compute episode lengths, and thus define a distribution
        # for sampling episodes with a probability proportional to their length
        self.episode_lengths = self.episode_ends - self.episode_starts + 1
        self.episode_probabilities = self.episode_lengths / sum(self.episode_lengths)

        # set state and action dimensions
        self.state_dim, self.act_dim = (
            np.prod(minari_dataset.get_observation_shape()),
            minari_dataset.get_action_size(),
        )

        # store observations, actions and rewards
        self.observations = minari_dataset.observations.reshape((-1, self.state_dim))
        self.actions = to_one_hot(minari_dataset.actions, self.act_dim) # convert from index integer representation to one-hot
        self.rewards = minari_dataset.rewards

        # compute observation statistics
        self.observation_mean, self.observation_std = (
            np.mean(self.observations, axis=0),
            np.std(self.observations, axis=0) + 1e-6,  # avoid division by zero
        )

    def _normalise_states(self, states):
        return (states - self.observation_mean) / self.observation_std

    # helper func to pad 2D or 3D numpy array along axis 1
    def _pad(self, x, pad_width=None, before=True, val=0):
        pad_width = pad_width or max(self.max_sample_len - x.shape[1], 0)
        pad_shape = [(0, 0)] * len(x.shape)
        pad_shape[1] = (pad_width, 0) if before else (0, pad_width)
        return np.pad(x, pad_shape, constant_values=val)

    def _discounted_cumsum(self, x, gamma):
        return np.array(list(accumulate(x[::-1], lambda a, b: (gamma * a) + b)))[::-1]

    def _sample_batch(self, batch_size):
        s, a, r, rtg, timesteps, mask = [], [], [], [], [], []

        # sample episodes with a probability proportional to their length
        episode_indices = np.random.choice(
            np.arange(self.num_episodes), size=batch_size, p=self.episode_probabilities
        )

        # sample a subsequence of each chosen episode
        for ep_idx in episode_indices:
            # sample an actual random start timestep for this episode
            # note: end represents the last timestep _included_ in the sequence,
            # which means when we're using exclusive range operators like [:]
            # or np.arange, we need to use end + 1
            start = np.random.randint(self.episode_starts[ep_idx], self.episode_ends[ep_idx])
            end = min(start + self.max_sample_len - 1, self.episode_ends[ep_idx])

            # store data
            timesteps.append(self._pad(np.arange(start, end + 1).reshape(1, -1)))
            s.append(
                self._normalise_states(
                    self._pad(
                        self.observations[start : end + 1].reshape(1, -1, self.state_dim)
                    )
                )
            )
            a.append(self._pad(self.actions[start : end + 1].reshape(1, -1, self.act_dim), val=-10))
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
            "states": torch.from_numpy(np.concatenate(s, axis=0)).float(),
            "actions": torch.from_numpy(np.concatenate(a, axis=0)).float(),
            "rewards": torch.from_numpy(np.concatenate(r, axis=0)).float(),
            "returns_to_go": torch.from_numpy(np.concatenate(rtg, axis=0)).float(),
            "timesteps": torch.from_numpy(np.concatenate(timesteps, axis=0)).long(),
            "attention_mask": torch.from_numpy(np.concatenate(mask, axis=0)).float(),
        }

    def __call__(self, features):
        batch_size = len(features)
        return self._sample_batch(batch_size)


# @dataclass
# class DecisionTransformerGymDataCollator:
#     return_tensors: str = "pt"
#     max_sample_len: int = 20  # subsets of the episode we use for training
#     state_dim: int = 17  # size of state space
#     act_dim: int = 6  # size of action space
#     max_ep_len: int = 1000  # max episode length in the dataset
#     scale: float = 1000.0  # normalization of rewards/returns
#     state_mean: np.array = None  # to store state means
#     state_std: np.array = None  # to store state stds
#     p_sample: np.array = None  # a distribution to take account trajectory lengths
#     n_traj: int = 0  # to store the number of trajectories in the dataset

#     def __init__(self, dataset) -> None:
#         self.act_dim = len(dataset[0]["actions"][0])
#         self.state_dim = len(dataset[0]["observations"][0])
#         self.dataset = dataset
#         # calculate dataset stats for normalization of states
#         states = []  # list of states - length = total_timesteps (across all episodes)
#         traj_lens = []  # episode lengths
#         for obs in dataset["observations"]:
#             states.extend(obs)
#             traj_lens.append(len(obs))
#         self.n_traj = len(traj_lens)  # number of episodes
#         states = np.vstack(states)  # numpy array of shape (total_timesteps, state_dim)
#         self.state_mean, self.state_std = (
#             np.mean(states, axis=0),
#             np.std(states, axis=0) + 1e-6,
#         )

#         # calculate a distribution to account for trajectory lengths
#         # this is used to sample episodes with a probability proportional to their length
#         traj_lens = np.array(traj_lens)
#         self.p_sample = traj_lens / sum(traj_lens)

#     def _discount_cumsum(self, x, gamma):
#         discount_cumsum = np.zeros_like(x)
#         discount_cumsum[-1] = x[-1]
#         for t in reversed(range(x.shape[0] - 1)):
#             discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
#         return discount_cumsum

#     def __call__(self, features):
#         batch_size = len(features)

#         # sample a number batch_size of episode indices, with replacement
#         # and with probability proportional to the length of the episode
#         batch_inds = np.random.choice(
#             np.arange(self.n_traj),
#             size=batch_size,
#             replace=True,
#             p=self.p_sample,  # reweights so we sample according to timesteps
#         )

#         # a batch of dataset features
#         # each of these lists will have length batch_size,
#         # and each element will correspond to a single episode
#         s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []

#         # loop through sampled episode indices
#         for idx in batch_inds:
#             episode = self.dataset[int(idx)]

#             # sample a random start index for the episode
#             si = random.randint(0, len(episode["rewards"]) - 1)

#             # get sequences from dataset
#             s.append(  # states
#                 np.array(episode["observations"][si : si + self.max_sample_len]).reshape(
#                     1, -1, self.state_dim
#                 )  # shape (1, <= max_sample_len, state_dim)
#             )
#             a.append(  # actions
#                 np.array(episode["actions"][si : si + self.max_sample_len]).reshape(
#                     1, -1, self.act_dim
#                 )  # shape (1, <= max_sample_len, act_dim)
#             )
#             r.append(  # rewards
#                 np.array(episode["rewards"][si : si + self.max_sample_len]).reshape(1, -1, 1)
#             )  # shape (1, <= max_sample_len, 1)
#             # d.append(
#             #     np.array(episode["dones"][si : si + self.max_sample_len]).reshape(1, -1)
#             # )  # dones (shape (1, max_sample_len))

#             timesteps.append(
#                 np.arange(si, si + s[-1].shape[1]).reshape(1, -1)
#             )  # timesteps (shape (1, max_sample_len))
#             timesteps[-1][timesteps[-1] >= self.max_ep_len] = (
#                 self.max_ep_len - 1
#             )  # padding cutoff

#             # return-to-go: discounted cumulative sum of future rewards for each timestep
#             rtg.append(
#                 self._discount_cumsum(np.array(episode["rewards"][si:]), gamma=1.0)[
#                     : s[-1].shape[1]  # TODO check the +1 removed here
#                 ].reshape(1, -1, 1)
#             )  # shape (1, max_sample_len, 1) (same as rewards)
#             if rtg[-1].shape[1] < s[-1].shape[1]:
#                 print("if true")
#                 rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

#             # padding and state + reward normalization
#             tlen = s[-1].shape[1]  # length of the trajectory
#             s[-1] = np.concatenate(
#                 [np.zeros((1, self.max_sample_len - tlen, self.state_dim)), s[-1]], axis=1
#             )  # front-pad with zeros to make trajectory have length max_sample_len
#             s[-1] = (s[-1] - self.state_mean) / self.state_std
#             a[-1] = np.concatenate(
#                 [np.ones((1, self.max_sample_len - tlen, self.act_dim)) * -10.0, a[-1]],
#                 axis=1,
#             )
#             r[-1] = np.concatenate(
#                 [np.zeros((1, self.max_sample_len - tlen, 1)), r[-1]], axis=1
#             )
#             # d[-1] = np.concatenate([np.ones((1, self.max_sample_len - tlen)) * 2, d[-1]], axis=1)
#             rtg[-1] = (
#                 np.concatenate(
#                     [np.zeros((1, self.max_sample_len - tlen, 1)), rtg[-1]], axis=1
#                 )
#                 / self.scale
#             )
#             timesteps[-1] = np.concatenate(
#                 [np.zeros((1, self.max_sample_len - tlen)), timesteps[-1]], axis=1
#             )
#             mask.append(
#                 np.concatenate(
#                     [np.zeros((1, self.max_sample_len - tlen)), np.ones((1, tlen))], axis=1
#                 )
#             )

#         s = torch.from_numpy(np.concatenate(s, axis=0)).float()
#         a = torch.from_numpy(np.concatenate(a, axis=0)).float()
#         r = torch.from_numpy(np.concatenate(r, axis=0)).float()
#         d = torch.from_numpy(np.concatenate(d, axis=0))
#         rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
#         timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).long()
#         mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()

#         return {
#             "states": s,
#             "actions": a,
#             "rewards": r,
#             "returns_to_go": rtg,
#             "timesteps": timesteps,
#             "attention_mask": mask,
#         }
