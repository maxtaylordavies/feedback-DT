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
        full=False,
        episode_dist="uniform",
    ) -> None:
        (
            self.dataset,
            self.args,
            self.scale,
            self.gamma,
            self.embedding_dim,
        ) = (
            custom_dataset,
            args,
            scale,
            gamma,
            embedding_dim,
        )
        self.feedback = self.args["use_feedback"]
        self.mission = self.args["use_mission"]
        self.context_length = self.args["context_length"]
        self.batch_size = self.args["batch_size"]
        self.randomise_starts = self.args["randomise_starts"]
        self.full=self.args["use_full_ep"]
        self.episode_dist=self.args["ep_dist"]
        self.sentence_embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-TinyBERT-L6-v2", device="cpu"
        )
        self.sentence_embedding_downsampler = torch.nn.AvgPool1d(
            int(768 / self.embedding_dim)
        )

        # null_emb = torch.tensor(np.random.random((1, self.embedding_dim)))
        # self._feedback_embeddings_cache = {
        #     f: null_emb for f in ["", "No feedback available."]
        # }
        self._feedback_embeddings_cache = {}

        # null_emb = torch.tensor(np.random.random((1, self.embedding_dim)))
        # self._mission_embeddings_cache = {
        #     m: null_emb for m in ["", "No mission available."]
        # }
        self._mission_embeddings_cache = {}

        self.dataset.load_shard()
        self.state_dim = self.dataset.state_dim
        self.act_dim = self.dataset.act_dim
        self.feedback_mode = self.args["feedback_mode"]
        self.mission_mode = self.args["mission_mode"]
        self.random_mode = self.args["random_mode"]

        random_sentence_generator = RandomFeedback(self.random_mode)
        self.random_sentences = random_sentence_generator.get_random_sentences()

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
        if type == "feedback" and self.feedback_mode == "random":
            sentences = self._replace_with_random(sentences)
        if type == "mission" and self.mission_mode == "random":
            sentences = self._replace_with_random(sentences)
        return self._get_sentence_embeddings(
            self._mission_embeddings_cache
            if type == "mission"
            else self._feedback_embeddings_cache,
            sentences,
        ).reshape(1, -1, self.embedding_dim)

    # helper func to pad 2D or 3D numpy array along axis 1
    def _pad(self, x, pad_width=None, before=True, val=0):
        pad_width = pad_width or (
            max(self.dataset.max_steps - x.shape[1], 0)
            if self.full
            else max(self.context_length - x.shape[1], 0)
        )
        pad_shape = [(0, 0)] * len(x.shape)
        pad_shape[1] = (pad_width, 0) if before else (0, pad_width)

        return np.pad(x, pad_shape, constant_values=val)

    def _count_samples_processed(self, batch):
        n_non_zero = int(torch.count_nonzero(batch["timesteps"]))
        n_first_timesteps = batch["timesteps"].shape[1]
        return n_non_zero + n_first_timesteps

    def _replace_with_random(self, sentence):
        def replace_with_random(x):
            return random.sample(self.random_sentences, 1)[0]

        vfunc = np.vectorize(replace_with_random)
        new_episode_feedback = vfunc(sentence)

        return new_episode_feedback

    def _sample_batch(self, batch_size, train=True):
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

        # load a random shard - if it doesn't contain any episodes, load another one
        # if we try 10 times and still don't have enough episodes, raise an exception
        num_eps = 0
        while num_eps < batch_size:
            tries, eps_per_shard = 0, 0
            while eps_per_shard == 0:
                self.dataset.load_shard()
                tries, eps_per_shard = tries + 1, self.dataset.num_episodes
                if tries > 10:
                    raise Exception(
                        f"Failed to load a shard with at least one episode after {tries} tries."
                    )

            # sample episode indices according to self.episode_dist
            episode_indices = self.dataset.sample_episode_indices(
                batch_size - num_eps, self.episode_dist
            )

            # sample a subsequence of each chosen episode
            length = self.context_length if not self.full else None
            for ep_idx in episode_indices:
                ep = self.dataset.sample_episode(
                    ep_idx,
                    gamma=self.gamma,
                    length=length,
                    random_start=self.randomise_starts,
                    feedback=self.feedback,
                    mission=self.mission,
                )

                # pad episode data to self.context_length and append to batch
                for k, v in ep.items():
                    if k in ["feedback", "mission"]:
                        v = self._pad(v, val=f"No {k} available.")
                        v = self.embed_sentences(v, type=k)
                        k += "_embeddings"
                    else:  # handle all other data - pad with zeros
                        v = self._pad(v, val=-100) if k == "actions" else self._pad(v)
                    batch[k].append(v)

            num_eps += len(episode_indices)


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
