import os
import shutil

import cv2


class Visualiser:
    def __init__(
        self,
        env,
        directory,
        filename,
        seed,
        auto_release=True,
        size=None,
        fps=30,
        rgb=True,
    ):
        self.env = env
        self.directory = directory
        self.path = os.path.join(self.directory, f"{filename}.mp4")
        self.auto_release = auto_release
        self.size = size
        self.active = True
        self.fps = fps
        self.rgb = rgb

        if self.size is None:
            self.env.reset(seed=seed)
            self.size = self.env.render().shape[:2][::-1]

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
        frame = self.env.render()
        self._writer.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if self.rgb else frame)

    def release(self):
        self._writer.release()

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        self._start()
        self._write(obs)
        return obs

    def get_env(self):
        return self.env

    def step(self, *args, **kwargs):
        data = self.env.step(*args, **kwargs)

        self._write(data[0])

        if self.auto_release and data[2]:
            self.release()

        return data

    def save_as(self, label):
        shutil.copy(self.path, os.path.join(self.directory, f"{label}.mp4"))


class AtariVisualiser(Visualiser):
    def __init__(
        self,
        env,
        directory,
        filename,
        seed,
        auto_release=True,
    ):
        super().__init__(
            env,
            directory,
            filename,
            seed,
            auto_release=auto_release,
            size=(84, 84),
            fps=30,
            rgb=True,
        )

    def _write(self, obs):
        if not self.active:
            return
        frame = obs.cpu().numpy().reshape((self.env.window,) + self.size)[-1]
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        self._writer.write(frame)
