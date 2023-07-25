from dataclasses import dataclass

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.dataset.custom_dataset import CustomDataset
from src.utils.utils import log


@dataclass
class Collator:
    def __init__(
        self,
        custom_dataset: CustomDataset,
        feedback=True,
        mission=True,
        context_length=64,
        scale=1,
        gamma=0.99,
        embedding_dim=128,
        randomise_starts=False,
        episode_dist="uniform",
    ) -> None:
        (
            self.dataset,
            self.feedback,
            self.mission,
            self.context_length,
            self.scale,
            self.gamma,
            self.embedding_dim,
            self.randomise_starts,
            self.episode_dist,
        ) = (
            custom_dataset,
            feedback,
            mission,
            context_length,
            scale,
            gamma,
            embedding_dim,
            randomise_starts,
            episode_dist,
        )
        self.sentence_embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-TinyBERT-L6-v2", device="cpu"
        )
        self.sentence_embedding_downsampler = torch.nn.AvgPool1d(
            int(768 / self.embedding_dim)
        )

        null_emb = torch.tensor(np.random.random((1, self.embedding_dim)))
        self._feedback_embeddings_cache = {f: null_emb for f in ["", "No feedback available"]}

        null_emb = torch.tensor(np.random.random((1, self.embedding_dim)))
        self._mission_embeddings_cache = {m: null_emb for m in ["", "No mission available"]}

        self.dataset.load_shard()
        self.state_dim = self.dataset.state_dim
        self.act_dim = self.dataset.act_dim

        self.reset_counter()

    def reset_counter(self):
        self.samples_processed = 0

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

    # MAX LOOKING INTO IMPLEMENTING PADDING WITH PYTORCH
    # helper func to pad 2D or 3D numpy array along axis 1
    def _pad(self, x, pad_width=None, before=True, val=0):
        pad_width = pad_width or max(self.context_length - x.shape[1], 0)
        pad_shape = [(0, 0)] * len(x.shape)
        pad_shape[1] = (pad_width, 0) if before else (0, pad_width)
        return np.pad(x, pad_shape, constant_values=val)

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
        episode_indices = self.dataset.sample_episode_indices(batch_size, self.episode_dist)

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
            self.samples_processed += np.prod(batch["timesteps"].shape)

        return batch

    def __call__(self, features):
        batch_size = len(features)
        return self._sample_batch(batch_size)
