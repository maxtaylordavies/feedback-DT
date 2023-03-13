import os

import cv2
import gymnasium as gym
import torch
import numpy as np


class Visualiser(gym.Wrapper):
    def __init__(
        self,
        env,
        directory,
        filename,
        seed,
        auto_release=True,
        size=None,
        fps=None,
        rgb=True,
    ):
        super(Visualiser, self).__init__(env)
        self.directory = directory
        self.path = os.path.join(self.directory, f"{filename}.mp4")
        self.auto_release = auto_release
        self.active = True
        self.rgb = rgb

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

        if size is None:
            self.env.reset(seed=seed)
            self.size = self.env.render().shape[:2][::-1]
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
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self.path, fourcc, self.fps, self.size)

    def _write(self):
        if self.active:
            frame = self.env.render()
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
    


def visualise_model(args, collator, model, target_returns, num_repeats):
    # create the output directory if it doesn't exist
    output_dir = os.path.join(args["output"], args["run_name"])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # visualise the model policy for each target return
    for ret in target_returns:
        for rep in range(num_repeats):
            env = Visualiser(
                gym.make(args["env_name"], render_mode="rgb_array"),
                output_dir,
                filename=f"{ret}-{rep}",
                seed=args["seed"],
                fps=30,
            )
            _visualise_model(args, collator, model, env, target_return=ret)


def _visualise_model(
    args, collator, model, env, target_return=1000, one_hot=True, normalise=True
):
    max_ep_len, device = env.max_steps or 64, "cpu"
    model = model.to(device)

    state_mean = collator.state_mean.astype(np.float32)
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = collator.state_std.astype(np.float32)
    state_std = torch.from_numpy(state_std).to(device=device)

    episode_return, episode_length, tmp = 0, 0, env.reset(seed=args["seed"])

    target_return = torch.tensor(target_return, device=device, dtype=torch.float32).reshape(
        1, 1
    )
    states = (
        torch.from_numpy(tmp[0]["image"])
        .reshape(1, collator.state_dim)
        .to(device=device, dtype=torch.float32)
    )
    actions = torch.zeros((0, collator.act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    for t in range(max_ep_len):
        actions = torch.cat(
            [actions, torch.zeros((1, collator.act_dim), device=device)], dim=0
        )
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            states if not normalise else (states - state_mean) / state_std,
            actions,
            rewards,
            target_return,
            timesteps,
            one_hot=one_hot,
        )
        actions[-1] = action

        env_state, reward, done, _, _ = env.step(np.argmax(action))
        cur_state = (
            torch.from_numpy(env_state["image"])
            .to(device=device)
            .reshape(1, collator.state_dim)
        )
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        pred_return = target_return[0, -1] - reward
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)

        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)],
            dim=1,
        )

        episode_return += reward
        episode_length += 1

        if done:
            break


def visualise_training_episode(episode, idx, args, output_dir):
    env = Visualiser(
        gym.make(args["env_name"], render_mode="rgb_array"),
        output_dir,
        idx,
        seed=args["seed"],
        fps=30,
    )

    episode_return, _ = 0, env.reset(seed=args["seed"])

    for t in range(len(episode.actions)):
        _, reward, done, _, _ = env.step(episode.actions[t])
        episode_return += reward
        if done:
            break

    return episode_return
