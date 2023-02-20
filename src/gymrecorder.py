import os
import time

import cv2
import gym
from moviepy.editor import *


class Recorder(gym.Wrapper):
    def __init__(
        self,
        env,
        directory,
        filename="",
        auto_release=True,
        size=None,
        fps=None,
        rgb=True,
    ):
        super(Recorder, self).__init__(env)
        self.directory = directory
        self.filename = f"{time.time()}.mp4" if not filename else filename
        self.auto_release = auto_release
        self.active = True
        self.rgb = rgb

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

        if size is None:
            self.env.reset()
            self.size = self.env.render(mode="rgb_array").shape[:2][::-1]
        else:
            self.size = size

        if fps is None:
            if "video.frames_per_second" in self.env.metadata:
                self.fps = self.env.metadata["video.frames_per_second"]
            else:
                self.fps = 30
        else:
            self.fps = fps

    def pause(self):
        self.active = False

    def resume(self):
        self.active = True

    def _start(self):
        self.cliptime = time.time()
        self.path = f"{self.directory}/{self.cliptime}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        self._writer = cv2.VideoWriter(self.path, fourcc, self.fps, self.size)

    def _write(self):
        if self.active:
            frame = self.env.render(mode="rgb_array")
            if self.rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._writer.write(frame)

    def release(self):
        self._writer.release()

    def reset(self, *args, **kwargs):
        observation = self.env.reset(*args, **kwargs)
        self._start()
        self._write()
        return observation

    def step(self, *args, **kwargs):
        data = self.env.step(*args, **kwargs)
        self._write()

        if self.auto_release and data[2]:
            self.release()

        return data

    def save(self):
        clip = VideoFileClip(self.path)
        clip.write_videofile(
            os.path.join(self.directory, self.filename),
            progress_bar=False,
            verbose=False,
        )
