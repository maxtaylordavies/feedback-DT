import os

import cv2
import gymnasium as gym
import torch
import numpy as np
import pandas as pd
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from src.utils import discounted_cumsum, log


class EvaluationCallback(TrainerCallback):
    def __init__(
        self,
        user_args,
        collator,
        target_returns=[10, 100, 1000, 100000],
        num_repeats=10,
        gamma=1.0,
    ) -> None:
        super().__init__()
        self.user_args = user_args
        self.collator = collator
        self.target_returns = target_returns
        self.num_repeats = num_repeats
        self.gamma = gamma
        self.epochs_complete = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # create the output directory if it doesn't exist
        self.output_dir = os.path.join(self.user_args["output"], self.user_args["run_name"])
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # initialise a dataframe to store the results
        self.results = {"epoch": [], "return": [], "target_return": []}

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model,
        **kwargs,
    ):
        self.epochs_complete = state.epoch
        log(f"Running evaluation at end of epoch {self.epochs_complete}")
        self._evaluate_model(model)

    def _evaluate_model(self, model):
        # for each target return, run the model on the environment for a number of repeats
        # for each repeat, we save a video of the trajectory and record the episode return
        for ret in self.target_returns:
            for rep in range(self.num_repeats):
                env = Visualiser(
                    gym.make(self.user_args["env_name"], render_mode="rgb_array"),
                    self.output_dir,
                    filename=f"{ret}-{rep}",
                    seed=self.user_args["seed"],
                )
                self.results["epoch"].append(self.epochs_complete)
                self.results["target_return"].append(ret)
                self.results["return"].append(
                    self._run_model_on_environment(model, env, target_return=ret)
                )

                env.release()

        # write the returns data to disk
        pd.DataFrame(self.results).to_pickle(os.path.join(self.output_dir, "returns.pkl"))

    def _run_model_on_environment(self, model, env, target_return=1000):
        max_ep_len = env.max_steps or 64

        state_mean = torch.from_numpy(self.collator.state_mean.astype(np.float32)).to(
            device=self.device
        )
        state_std = torch.from_numpy(self.collator.state_std.astype(np.float32)).to(
            device=self.device
        )

        target_return = torch.tensor(
            target_return, device=self.device, dtype=torch.float32
        ).reshape(1, 1)

        tmp = env.reset(seed=self.user_args["seed"])

        states = (
            torch.from_numpy(tmp[0]["image"])
            .reshape(1, self.collator.state_dim)
            .to(device=self.device, dtype=torch.float32)
        )
        actions = torch.zeros(
            (0, self.collator.act_dim), device=self.device, dtype=torch.float32
        )
        rewards = torch.zeros(0, device=self.device, dtype=torch.float32)
        timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)

        for t in range(max_ep_len):
            actions = torch.cat(
                [actions, torch.zeros((1, self.collator.act_dim), device=self.device)], dim=0
            )
            rewards = torch.cat([rewards, torch.zeros(1, device=self.device)])

            actions[-1] = model.get_action(
                (states - state_mean) / state_std,
                actions,
                rewards,
                target_return,
                timesteps,
                one_hot=True,
            )
            a = actions[-1].detach().cpu().numpy()

            env_state, reward, done, _, _ = env.step(np.argmax(a))
            cur_state = (
                torch.from_numpy(env_state["image"])
                .to(device=self.device)
                .reshape(1, self.collator.state_dim)
            )
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            pred_return = target_return[0, -1] - reward
            target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)

            timesteps = torch.cat(
                [
                    timesteps,
                    torch.ones((1, 1), device=self.device, dtype=torch.long) * (t + 1),
                ],
                dim=1,
            )

            if done:
                break

        return discounted_cumsum(rewards.detach().cpu().numpy(), self.gamma)[0]


class Visualiser(gym.Wrapper):
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
        super(Visualiser, self).__init__(env)
        self.directory = directory
        self.path = os.path.join(self.directory, f"{filename}.mp4")
        self.auto_release = auto_release
        self.active = True
        self.fps = fps
        self.rgb = rgb

        if size is None:
            self.env.reset(seed=seed)
            self.size = self.env.render().shape[:2][::-1]
        else:
            self.size = size

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
