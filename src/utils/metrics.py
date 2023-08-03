import os
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
from jsonc_parser.parser import JsoncParser

from src.dataset import MinariDataset
from src.dataset.custom_feedback_verifier import TaskFeedback


class EvaluationMetric(ABC):
    def __init__(self, level, test_episodes_path=None):
        self.level = level
        self.test_episodes_path = test_episodes_path

    @abstractmethod
    def calculate(self):
        raise NotImplementedError


class PathLength(EvaluationMetric):
    def __init__(self, level, test_episodes_path=None):
        super().__init__(level, test_episodes_path)

    def calculate(self):
        """
        Calculate the mean and std for the number of steps for all test episodes.

        Returns
        -------
            n_mean (float): mean number of steps per episode.
            n_std (float): std of the number of steps per episode.
        """
        dataset = MinariDataset.load(self.test_episodes_path)
        path_lengths = []
        for episode in dataset.episodes:
            path_lengths.append(len(episode))
        return np.mean(path_lengths), np.std(path_lengths)


class Reward(EvaluationMetric):
    def __init__(self, level, test_episodes_path=None):
        super().__init__(level, test_episodes_path)

    def calculate(self):
        """
        Calculate the mean and std for the reward for all test episode.

        Returns
        -------
            r_mean (float): mean reward per episode.
            r_std (float): std of the reward per episode.
        """
        dataset = MinariDataset.load(self.test_episodes_path)
        rewards = []
        for episode in dataset.episodes:
            rewards.append(episode[-1]["reward"])
        return np.mean(rewards), np.std(rewards)


class SuccessRate(EvaluationMetric):
    def __init__(self, level, test_episodes_path=None):
        super().__init__(level, test_episodes_path)

    def calculate(self):
        """
        Calculate the mean and std for the success rate for all test episodes.

        Success rate is defined as the number of episodes that reached the goal (were terminated)
        divided by the total number of episodes.

        The calculation relies on the termination flag.

        Returns
        -------
            sr_mean (float): mean success rate per episode.
            sr_std (float): std of the success rate per episode.
        """
        dataset = MinariDataset.load(self.test_episodes_path)
        success = []
        for episode in dataset.episodes:
            if episode[-1]["termination"]:
                success.append(1)
            else:
                success.append(0)
        return np.mean(success), np.std(success)


class PWSuccessRate(EvaluationMetric):
    def __init__(self, level, test_episodes_path=None):
        super().__init__(level, test_episodes_path)
        self.demo_mean = self._get_demo_mean()

    def _get_demo_mean(self):
        metadata_path = os.getenv("ENV_METADATA_PATH", "env_metadata.jsonc")
        metadata = JsoncParser.parse_file(metadata_path)["levels"]
        for level_group, levels in metadata.items():
            if self.level in levels:
                return metadata[level_group][self.level]["demo_mean_n_steps"]

    def _get_pw_success(self, episode):
        return episode[-1]["termination"] * (
            len(episode) / max(len(episode), self.demo_mean)
        )

    def calculate(self):
        """
        Calculate the mean and std for the success rate for all test episodes, weighted by the path length.

        Success rate is defined as the number of episodes that reached the goal (were terminated)
        divided by the total number of episodes.

        The calculation relies on:
         - the termination flag,
         - the path length for test episode and
         - the average path length for demonstrations (from original BabyAI paper, see metadata).
        """
        dataset = MinariDataset.load(self.test_episodes_path)
        success = []
        for episode in dataset.episodes:
            if episode[-1]["termination"]:
                success.append(self._get_pw_success(episode))
            else:
                success.append(0)
        return np.mean(success), np.std(success)


class GCSuccessRate(EvaluationMetric):
    def __init__(self, level, test_episodes_path=None):
        super().__init__(level, test_episodes_path)
        self.dataset = MinariDataset.load(self.test_episodes_path)
        self.n_gcs = self._get_n_gcs()

    def _get_n_gcs(self):
        n_gcs = []
        for episode in self.dataset.episodes:
            env = gym.make(episode["config"])
            env.reset(episode["seed"])
            task_feedback_verifier = TaskFeedback(env)
            n_gcs.append(len(task_feedback_verifier.subtasks))
        return n_gcs

    def calculate(self):
        """
        Calculate the mean and std for the goal condition success rate for all test episodes.

        Goal condition success rate is defined as the number of goal conditions that were met
        (sub-goals that were achieved) divided by the total number of goal conditions.

        Note that this requires the test episodes to have been created with task_only feedback.

        The calculation relies on:
         - goal condition flags (using the same logic as for numerical rewards for task success feedback)
         - the gold standard number of goal conditions (TBC)
        """
        dataset = MinariDataset.load(self.test_episodes_path)
        gc_successes = []
        for episode in dataset.episodes:
            n_gcs_met = 0
            for step in episode:
                if step["feedback"] != "No feedback available.":
                    n_gcs_met += 1
            gc_successes.append(n_gcs_met)
        return np.mean((gc_successes / self.n_gcs)), np.std((gc_successes / self.n_gcs))
