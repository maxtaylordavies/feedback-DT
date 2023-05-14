import os

import cv2
import gymnasium as gym
import torch
import numpy as np
import pandas as pd
from minigrid.wrappers import FullyObsWrapper
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
import matplotlib.pyplot as plt
import seaborn as sns

from src.agent import Agent, AgentInput
from src.utils.utils import discounted_cumsum, log

sns.set_theme()


class Evaluator(TrainerCallback):
    def __init__(
        self,
        user_args,
        collator,
        sample_interval=100,
        target_return=1000,
        num_repeats=10,
        gamma=1.0,
    ) -> None:
        super().__init__()
        self.user_args = user_args
        self.collator = collator
        self.sample_interval = sample_interval
        self.target_return = target_return
        self.num_repeats = num_repeats
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # create the output directory if it doesn't exist
        self.output_dir = os.path.join(self.user_args["output"], self.user_args["run_name"])
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # initialise a dataframe to store the results
        self.results = {"samples": [], "return": []}

        # create blank feedback embeddings
        self.feedback_embeddings = self.collator._embed_feedback(
            np.array([[""] * user_args["context_length"]]).reshape(-1, 1)
        ).to(self.device)

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        if not self.user_args["plot_on_train_end"]:
            return

        # convert results to dataframe
        df = pd.DataFrame(self.results)
        sns.lineplot(x="samples", y="return", data=df)
        plt.show()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Agent,
        **kwargs,
    ):
        prev = self.results["samples"][-1] if len(self.results["samples"]) else 0
        if self.collator.samples_processed - prev >= self.sample_interval:
            log(
                f"Running evaluation (samples: {self.collator.samples_processed}, epoch: {state.epoch}, step: {state.global_step})",
                with_tqdm=True,
            )
            self._evaluate_agent(model)

    def _evaluate_agent(self, agent: Agent):
        # for each repeat, record the episode return (and optionally render a video of the episode)
        for rep in range(self.num_repeats):
            _env = gym.make(self.user_args["env_name"], render_mode="rgb_array")
            env = (
                Visualiser(
                    _env, self.output_dir, filename=f"{rep}", seed=self.user_args["seed"]
                )
                if self.user_args["record_video"]
                else _env
            )

            self.results["samples"].append(self.collator.samples_processed)
            self.results["return"].append(self._run_agent_on_environment(agent, env))

            if self.user_args["record_video"]:
                env.release()

        # convert results to dataframe
        df = pd.DataFrame(self.results)

        # log the average episode return for the current eval
        log(
            f"Average episode return: {df[df['samples'] == self.collator.samples_processed]['return'].mean()}",
            with_tqdm=True,
        )

        # save the results to disk
        df.to_pickle(os.path.join(self.output_dir, "returns.pkl"))

    def _run_agent_on_environment(self, agent: Agent, env: gym.Env):
        max_ep_len = env.max_steps if hasattr(env, "max_steps") else 100

        state_mean = torch.from_numpy(self.collator.state_mean.astype(np.float32)).to(
            device=self.device
        )
        state_std = torch.from_numpy(self.collator.state_std.astype(np.float32)).to(
            device=self.device
        )

        obs, _ = env.reset(seed=self.user_args["seed"])
        # fully_obs_env = FullyObsWrapper(env)
        # obs = fully_obs_env.observation(tmp[0])

        states = (
            torch.from_numpy(obs)
            .reshape(1, self.collator.state_dim)
            .to(device=self.device, dtype=torch.float32)
        )
        actions = torch.zeros(
            (0, self.collator.act_dim), device=self.device, dtype=torch.float32
        )
        rewards = torch.zeros(0, device=self.device, dtype=torch.float32)
        returns_to_go = torch.tensor(
            self.target_return, device=self.device, dtype=torch.float32
        ).reshape(1, 1)
        timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)

        for t in range(max_ep_len):
            actions = torch.cat(
                [actions, torch.zeros((1, self.collator.act_dim), device=self.device)], dim=0
            )
            rewards = torch.cat([rewards, torch.zeros(1, device=self.device)])

            actions[-1] = agent.get_action(
                AgentInput(
                    states=(states - state_mean) / state_std,
                    actions=actions,
                    rewards=rewards,
                    returns_to_go=returns_to_go,
                    timesteps=timesteps,
                    feedback_embeddings=self.feedback_embeddings,
                    attention_mask=None,
                ),
                context=self.user_args["context_length"],
                one_hot=True,
            )
            a = actions[-1].detach().cpu().numpy()

            obs, reward, done, _, _ = env.step(np.argmax(a))
            # obs = fully_obs_env.observation(obs)
            cur_state = (
                torch.from_numpy(obs)
                .to(device=self.device)
                .reshape(1, self.collator.state_dim)
            )
            states = torch.cat([states, cur_state], dim=0)

            rewards[-1] = reward
            pred_return = returns_to_go[0, -1] - reward
            returns_to_go = torch.cat([returns_to_go, pred_return.reshape(1, 1)], dim=1)

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
