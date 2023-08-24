import gymnasium as gym
import numpy as np


class FeedbackEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.task_fv = None
        self.rule_fv = None
        self.feedback_mode = None

    def setup_feedback(self, feedback_mode, random_fv, task_fv, rule_fv):
        self.feedback_mode = feedback_mode
        self.random_fv = random_fv
        self.task_fv = task_fv
        self.rule_fv = rule_fv

    def rule_feedback(self, action):
        return (
            self.rule_fv.verify_feedback(self, action)
            if self.feedback_mode in ["all", "rule_only"]
            else None
        )

    def task_feedback(self, action):
        return (
            self.task_fv.verify_feedback(self, action)
            if self.feedback_mode in ["all", "task_only"]
            else None
        )

    def get_feedback(self, rule_feedback, task_feedback):
        if self.feedback_mode == "random":
            return self.random_fv.verify_feedback()
        if self.feedback_mode == "rule_only":
            return rule_feedback
        if self.feedback_mode == "task_only":
            return task_feedback
        if self.feedback_mode == "numerical_reward":
            if task_feedback != "No feedback available.":
                return np.array(1)
            if rule_feedback != "No feedback available.":
                return np.array(-1)
            return np.array(0)
        if self.feedback_mode == "all":
            if rule_feedback == "No feedback available.":
                return task_feedback
            return rule_feedback
        raise ValueError(f"Unknown feedback mode: {self.feedback_mode}")

    def step(self, action):
        if not self.feedback_mode:
            obs, reward, terminated, truncated, _ = super().step(action)
            return obs, reward, terminated, truncated, None

        # get rule feedback (before taking action)
        rule_feedback = self.rule_feedback(action)

        # call parent step
        obs, reward, terminated, truncated, _ = super().step(action)

        # get task feedback (after taking action)
        task_feedback = self.task_feedback(action)

        # get feedback + return tuple
        feedback = self.get_feedback(rule_feedback, task_feedback)
        return obs, reward, terminated, truncated, feedback

    @classmethod
    def from_env(cls, env, feedback_mode, verifiers):
        new_env = cls()
        new_env.__dict__ = env.__dict__.copy()
        new_env.setup_feedback(
            feedback_mode,
            verifiers["random"],
            verifiers["task"],
            verifiers["rule"],
        )
        return new_env
