import os
import shutil
from typing import Optional

import cv2
import gymnasium as gym

from .feedback_env import FeedbackEnv


class RecorderEnv(FeedbackEnv):
    def __init__(
        self,
        env: gym.Env,
        feedback_mode: Optional[str],
        directory,
        filename,
        auto_release=True,
        size=None,
        fps=30,
        rgb=True,
        max_steps=None,
    ):
        super().__init__(env, feedback_mode, max_steps)
        self.directory = os.path.join(directory, "recordings")
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.path = os.path.join(self.directory, f"{filename}.mp4")
        self.auto_release = auto_release
        self.size = size
        self.active = True
        self.fps = fps
        self.rgb = rgb

        if self.size is None:
            self.reset()
            self.size = self.render().shape[:2][::-1]

    def pause(self):
        self.active = False

    def resume(self):
        self.active = True

    def _start(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self.path, fourcc, self.fps, self.size)

    def _write(self, obs=None):
        if not self.active:
            return
        frame = self.render()
        self._writer.write(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if self.rgb else frame
        )

    def release(self):
        self._writer.release()

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self._start()
        self._write(obs)
        return obs, info

    def step(self, *args, **kwargs):
        data = super().step(*args, **kwargs)
        self._write(data[0])
        if self.auto_release and data[2]:
            self.release()
        return data

    def save_as(self, label):
        shutil.copy(self.path, os.path.join(self.directory, f"{label}.mp4"))
