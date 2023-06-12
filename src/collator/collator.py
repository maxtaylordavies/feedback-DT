from dataclasses import dataclass

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.dataset.minari_dataset import MinariDataset
from src.utils.utils import discounted_cumsum, to_one_hot


@dataclass
class Collator:
    def __init__(
        self,
        custom_dataset: MinariDataset,
        feedback: True,
        context_length=64,
        scale=1,
        gamma=1.0,
        embedding_dim=128,
        randomise_starts=False,
    ) -> None:
        (
            self.context_length,
            self.scale,
            self.gamma,
            self.embedding_dim,
            self.randomise_starts,
        ) = (
            context_length,
            scale,
            gamma,
            embedding_dim,
            randomise_starts,
        )

        # compute start and end timesteps for each episode
        self.episode_ends = np.where(
            custom_dataset.terminations + custom_dataset.truncations == 1
        )[0]
        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1] + 1])
        self.num_episodes = len(self.episode_starts)

        # define a distribution for sampling episodes with probability inversely proportional to their length
        self.episode_probabilities = 1 / (self.episode_ends - self.episode_starts + 1)
        self.episode_probabilities /= np.sum(self.episode_probabilities)

        # set state and action dimensions
        self.state_dim, self.act_dim = (
            np.prod(custom_dataset.observations.shape[1:]),
            custom_dataset.get_action_size(),
        )

        # store observations, actions and rewards
        self.observations = custom_dataset.observations.reshape((-1, self.state_dim))
        self.actions = to_one_hot(
            custom_dataset.actions, width=self.act_dim
        )  # convert from index integer representation to one-hot
        self.rewards = custom_dataset.rewards

        # compute observation statistics
        self.state_mean, self.state_std = (
            np.mean(self.observations, axis=0),
            np.std(self.observations, axis=0) + 1e-6,  # avoid division by zero
        )

        # store feedback as flattened array. if no feedback provided, use empty strings
        self.feedback = (
            np.hstack(custom_dataset.feedback)
            if feedback
            else np.array(["no feedback available"] * len(self.observations))
        )
        self._feedback_embeddings_map = (
            self._precompute_feedback_embeddings()
            if feedback
            else {"": torch.tensor(np.random.random((1, self.embedding_dim)))}
        )

        self.reset_counter()

    def reset_counter(self):
        self.samples_processed = 0

    def _precompute_feedback_embeddings(self):
        model = SentenceTransformer(
            "sentence-transformers/paraphrase-TinyBERT-L6-v2", device="cpu"
        )
        downsampler = torch.nn.AvgPool1d(int(768 / self.embedding_dim))

        # create a mapping from each unique feedback string to an embedding of shape (1, embedding_dim)
        return {
            s: downsampler(
                model.encode(s, convert_to_tensor=True, device="cpu").reshape(1, -1)
            )
            for s in np.unique(
                np.append(self.feedback, "")
            )  # add empty string to ensure it has an embedding
        }

    def _embed_feedback(self, feedback):
        emb = torch.zeros((len(feedback), self.embedding_dim))
        for i in range(len(feedback)):
            emb[i] = self._feedback_embeddings_map[feedback[i, 0]]
        return emb

    def _normalise_states(self, states):
        return (states - self.state_mean) / self.state_std

    # helper func to pad 2D or 3D numpy array along axis 1
    def _pad(self, x, pad_width=None, before=True, val=0):
        pad_width = pad_width or max(self.context_length - x.shape[1], 0)
        pad_shape = [(0, 0)] * len(x.shape)
        pad_shape[1] = (pad_width, 0) if before else (0, pad_width)
        return np.pad(x, pad_shape, constant_values=val)

    def _sample_batch(self, batch_size):
        t, s, a, r, f, rtg, mask = [], [], [], [], [], [], []

        # sample episodes with a probability inversely proportional to their length (successful eps are shorter)
        episode_indices = np.random.choice(
            np.arange(self.num_episodes), size=batch_size, p=self.episode_probabilities
        )

        # sample a subsequence of each chosen episode
        for ep_idx in episode_indices:
            # optionally sample a random start timestep for this episode
            # note: end represents the last timestep _included_ in the sequence,
            # which means when we're using exclusive range operators like [:]
            # or np.arange, we need to use end + 1
            start = (
                np.random.randint(
                    self.episode_starts[ep_idx], self.episode_ends[ep_idx]
                )
                if self.randomise_starts
                else self.episode_starts[ep_idx]
            )
            end = min(start + self.context_length - 1, self.episode_ends[ep_idx])

            # increment counter
            self.samples_processed += end - start + 1

            # timesteps
            t.append(self._pad(np.arange(0, end - start + 1).reshape(1, -1)))

            # states
            s.append(
                self._normalise_states(
                    self._pad(
                        self.observations[start : end + 1].reshape(
                            1, -1, self.state_dim
                        )
                    )
                )
            )

            # actions
            a.append(
                self._pad(
                    self.actions[start : end + 1].reshape(1, -1, self.act_dim), val=-10
                )
            )

            # rewards
            r.append(self._pad(self.rewards[start : end + 1].reshape(1, -1, 1)))

            # returns-to-go
            rtg.append(
                self._pad(
                    discounted_cumsum(
                        self.rewards[start : self.episode_ends[ep_idx] + 1],
                        gamma=self.gamma,
                    )[: s[-1].shape[1]].reshape(1, -1, 1)
                )
                / self.scale
            )

            # feedback
            f.append(
                self._embed_feedback(
                    self._pad(self.feedback[start : end + 1].reshape(1, -1, 1), val="")[
                        0
                    ]
                ).reshape(1, -1, self.embedding_dim)
            )

            # attention mask
            mask.append(
                np.concatenate(
                    [
                        np.zeros((1, self.context_length - (end - start + 1))),
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
            "feedback_embeddings": torch.cat(f, axis=0),
        }

    def __call__(self, features):
        batch_size = len(features)
        return self._sample_batch(batch_size)
