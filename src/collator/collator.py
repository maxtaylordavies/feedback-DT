import random
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import WeightedRandomSampler

from src.constants import GLOBAL_SEED
from src.dataset.custom_dataset import CustomDataset
from src.dataset.custom_feedback_verifier import RandomFeedback
from src.utils.utils import log


@dataclass
class Collator:
    def __init__(
        self,
        custom_dataset: CustomDataset,
        args,
        scale=1,
        gamma=0.99,
        embedding_dim=128,
        randomise_starts=False,
        episode_dist="uniform",
    ) -> None:
        (
            self.dataset,
            self.args,
            self.scale,
            self.gamma,
            self.embedding_dim,
            self.randomise_starts,
            self.episode_dist,
        ) = (
            custom_dataset,
            args,
            scale,
            gamma,
            embedding_dim,
            randomise_starts,
            episode_dist,
        )
        self.feedback = self.args["use_feedback"]
        self.mission = self.args["use_mission"]
        self.context_length = self.args["context_length"]
        self.batch_size = self.args["batch_size"]
        self.sentence_embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-TinyBERT-L6-v2", device="cpu"
        )
        self.sentence_embedding_downsampler = torch.nn.AvgPool1d(
            int(768 / self.embedding_dim)
        )

        null_emb = torch.tensor(np.random.random((1, self.embedding_dim)))
        self._feedback_embeddings_cache = {
            f: null_emb for f in ["", "No feedback available."]
        }

        null_emb = torch.tensor(np.random.random((1, self.embedding_dim)))
        self._mission_embeddings_cache = {
            m: null_emb for m in ["", "No mission available."]
        }

        self.dataset.load_shard()
        self.state_dim = self.dataset.state_dim
        self.act_dim = self.dataset.act_dim
        self.feedback_mode = self.args["feedback_mode"]

        if self.feedback and "random" in self.feedback_mode:
            random_feedback_generator = RandomFeedback(self.feedback_mode)
            self.random_feedback_sentences = (
                random_feedback_generator.get_random_sentences()
            )

        self.reset_counter()

    def reset_counter(self):
        self.samples_processed = 0

    def update_epoch(self):
        pass

    def _compute_sentence_embedding(self, sentence):
        return self.sentence_embedding_downsampler(
            self.sentence_embedding_model.encode(
                sentence, convert_to_tensor=True, device="cpu"
            ).reshape(1, -1)
        )

    # embed sentences (with cache)
    def _get_sentence_embeddings(self, embeddings_cache, sentences):
        sentences = sentences.flatten()
        emb = torch.zeros((len(sentences), self.embedding_dim))
        for i in range(len(sentences)):
            s = sentences[i]
            if s not in embeddings_cache:
                embeddings_cache[s] = self._compute_sentence_embedding(s)
            emb[i] = embeddings_cache[s]
        return emb

    def embed_sentences(self, sentences, type):
        if type not in ("feedback", "mission"):
            raise Exception(f"Got unsupported sentence type: {type}")
        return self._get_sentence_embeddings(
            self._mission_embeddings_cache
            if type == "mission"
            else self._feedback_embeddings_cache,
            sentences,
        ).reshape(1, -1, self.embedding_dim)

    # helper func to pad 2D or 3D numpy array along axis 1
    def _pad(self, x, pad_width=None, before=True, val=0):
        pad_width = pad_width or max(self.context_length - x.shape[1], 0)
        pad_shape = [(0, 0)] * len(x.shape)
        pad_shape[1] = (pad_width, 0) if before else (0, pad_width)
        return np.pad(x, pad_shape, constant_values=val)

    def _count_samples_processed(self, batch):
        n_non_zero = int(torch.count_nonzero(batch["timesteps"]))
        n_first_timesteps = batch["timesteps"].shape[1]
        return n_non_zero + n_first_timesteps

    def _replace_with_random(self, epsiode_feedback):
        def replace_with_random(x):
            if x == "No feedback available.":
                return (
                    "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
                    if "lorem" in self.feedback_mode
                    else x
                )
            return random.sample(self.random_feedback_sentences, 1)[0]

        vfunc = np.vectorize(replace_with_random)
        new_episode_feedback = vfunc(epsiode_feedback)

        return new_episode_feedback

    def _sample_batch(self, batch_size, random_start=True, full=False, train=True):
        batch = {
            "timesteps": [],
            "mission_embeddings": [],
            "states": [],
            "actions": [],
            "rewards": [],
            "returns_to_go": [],
            "attention_mask": [],
            "feedback_embeddings": [],
        }

        # first, load a random shard from file into the CustomDataset object
        self.dataset.load_shard()

        # sample episode indices according to self.episode_dist
        episode_indices = self.dataset.sample_episode_indices(
            batch_size, self.episode_dist
        )

        # sample a subsequence of each chosen episode
        length = self.context_length if not full else None
        for ep_idx in episode_indices:
            ep = self.dataset.sample_episode(
                ep_idx,
                gamma=self.gamma,
                length=length,
                random_start=random_start,
                feedback=self.feedback,
                mission=self.mission,
            )

            # pad episode data to self.context_length and append to batch
            for k, v in ep.items():
                if k == "feedback":
                    if "random" in self.feedback_mode:
                        v = self._replace_with_random(v)
                if k in (
                    "mission",
                    "feedback",
                ):  # handle sentence data - pad with empty strings and embed
                    v = self.embed_sentences(self._pad(v, val=""), type=k)
                    k += "_embeddings"
                else:  # handle all other data - pad with zeros
                    v = self._pad(v)
                batch[k].append(v)

        # convert batch to (concatenated) tensors
        for k, v in batch.items():
            if "embeddings" in k:
                batch[k] = torch.cat(v, axis=0)
            else:
                batch[k] = torch.from_numpy(np.concatenate(v, axis=0))
                batch[k] = batch[k].long() if k == "timesteps" else batch[k].float()

        # if we're in training mode, update the sample counter
        if train:
            self.samples_processed += self._count_samples_processed(batch)

        return batch

    def __call__(self, features):
        return self._sample_batch(self.batch_size)


class RoundRobinCollator:
    """
    Class to implement round-robin learning by cycling through a list of collators.

    Args:
        datasets (list): list of CustomDataset objects to cycle through.

    Returns:
        Collator: a collator object that samples a batch from a different dataset each time it is called.
    """

    def __init__(self, custom_dataset, args):
        self.datasets = custom_dataset
        self.args = args
        self.batch_size = self.args["batch_size"]
        self.collators = [Collator(dataset, self.args) for dataset in self.datasets]
        random.shuffle(self.collators)
        self.collator_idx = 0
        self.reset_counter()
        self.dataset = self.datasets[0]
        self.state_dim = self.dataset.state_dim
        self.act_dim = self.dataset.act_dim

    def reset_counter(self):
        self.samples_processed = 0
        self.samples_processed_by_level = {key: 0 for key in range(len(self.collators))}

    def update_epoch(self):
        pass

    def _count_samples_processed(self, batch):
        n_non_zero = int(torch.count_nonzero(batch["timesteps"]))
        n_first_timesteps = batch["timesteps"].shape[1]
        return n_non_zero + n_first_timesteps

    def _sample_batch(self, batch_size, train=True):
        collator = self.collators[self.collator_idx]
        features = np.zeros(batch_size)
        batch = collator(features)
        if train:
            self.samples_processed_by_level[
                self.collator_idx
            ] += self._count_samples_processed(batch)
            self.samples_processed += self._count_samples_processed(batch)

        self.collator_idx = (self.collator_idx + 1) % len(self.collators)

        return batch

    def __call__(self, features):
        return self._sample_batch(self.batch_size)


class CurriculumCollator:
    """
    Class to implement curriculum learning by composing weighted batches from a list of collators.
    """

    def __init__(self, custom_dataset, args):
        self.datasets = custom_dataset
        self.args = args
        self.custom_order = (
            [int(n) for n in args["custom_order"].split(",")]
            if args["custom_order"]
            else None
        )
        if self.custom_order:
            self._custom_sort()
        self.anti = "anti" in args["train_mode"]
        self.collators = (
            [Collator(dataset, self.args) for dataset in self.datasets]
            if not self.anti
            else [Collator(dataset, self.args) for dataset in reversed(self.datasets)]
        )

        self.reset_counter()
        self.reset_weights()
        self.dataset = self.datasets[0]
        self.state_dim = self.dataset.state_dim
        self.act_dim = self.dataset.act_dim
        self.generator = torch.Generator().manual_seed(GLOBAL_SEED)

    def _custom_sort(self):
        zipped_pairs = zip(self.custom_order, self.datasets)
        self.datasets = [dataset for _, dataset in sorted(zipped_pairs)]

    def reset_counter(self):
        self.samples_processed = 0
        self.samples_processed_by_level = {key: 0 for key in range(len(self.datasets))}

    def reset_weights(self):
        self.weights = np.zeros(len(self.collators))
        self.weights[0] = 1

    def update_epoch(self):
        self._update_weights()

    def _update_weights(self):
        n_tasks_to_include = np.argmax(self.weights) + 2
        triangle = n_tasks_to_include * (n_tasks_to_include + 1) / 2
        if n_tasks_to_include <= len(self.collators):
            for idx in range(n_tasks_to_include):
                self.weights[idx] = (idx + 1) / triangle
            # log(f"weights updated to:\n{self.weights}")

    def _count_samples_processed(self, batch, n_samples=None):
        if n_samples is None:
            n_non_zero = int(torch.count_nonzero(batch["timesteps"]))
        else:
            n_non_zero = int(torch.count_nonzero(batch["timesteps"][:n_samples]))
        n_first_timesteps = batch["timesteps"].shape[1]
        return n_non_zero + n_first_timesteps

    def _sample_batch(self, batch_size, train=True):
        features = np.zeros(batch_size)
        sampler = WeightedRandomSampler(
            self.weights, batch_size, generator=self.generator
        )
        sample_ids = sorted(list(sampler))

        mixed_batch = {}
        for dataset_idx, n_samples in Counter(sample_ids).items():
            batch = self.collators[dataset_idx](features)
            for key, tensor in batch.items():
                try:
                    mixed_batch[key] = torch.cat(
                        (mixed_batch[key], tensor[:n_samples]), dim=0
                    )
                except KeyError:
                    mixed_batch[key] = tensor[:n_samples]
            self.samples_processed_by_level[
                dataset_idx
            ] += self._count_samples_processed(batch, n_samples)

        if train:
            self.samples_processed += self._count_samples_processed(mixed_batch)
        # log(self.samples_processed_by_level)

        return mixed_batch

    def __call__(self, features):
        batch_size = self.args["batch_size"]
        return self._sample_batch(batch_size)
