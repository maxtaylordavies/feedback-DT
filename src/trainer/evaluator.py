import os
import shutil

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

from src.agent import Agent, AgentInput, RandomAgent
from src.utils.utils import discounted_cumsum, log
from .atari_env import AtariEnv

sns.set_theme()


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
        obs = self.env.reset(*args, **kwargs)
        self._start()
        self._write(obs)
        return obs

    def step(self, *args, **kwargs):
        data = self.env.step(*args, **kwargs)

        self._write()

        if self.auto_release and data[2]:
            self.release()

        return data

    def save_as_best(self):
        shutil.copy(self.path, os.path.join(self.directory, "best.mp4"))


class AtariVisualiser(Visualiser):
    def _write(self, obs):
        if not self.active:
            return
        frame = obs.numpy().reshape((self.env.window,) + self.size)[-1]
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        self._writer.write(frame)


class Evaluator(TrainerCallback):
    def __init__(
        self,
        user_args,
        collator,
        sample_interval=100,
        target_returns=[0, 90, 1000],
        num_repeats=10,
        gamma=0.99,
    ) -> None:
        super().__init__()
        self.user_args = user_args
        self.collator = collator
        self.sample_interval = sample_interval
        self.samples_processed = 0
        self.target_returns = target_returns
        self.best_return = -np.inf
        self.num_repeats = num_repeats
        self.gamma = gamma
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        # create the output directory if it doesn't exist
        self.output_dir = os.path.join(self.user_args["output"], self.user_args["run_name"])
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # initialise a dataframe to store the results
        self.results = {"samples": [], "return": [], "episode length": [], "model": []}

        # create blank feedback embeddings
        self.feedback_embeddings = self.collator._embed_feedback(
            np.array([[""] * user_args["context_length"]]).reshape(-1, 1)
        ).to(self.device)

        # create a random agent to evaluate against
        self.random_agent = RandomAgent(self.collator.act_dim)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Agent,
        **kwargs,
    ):
        # self._plot_loss(state)
        self._run_eval_and_plot(model, state, final=True)

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Agent,
        **kwargs,
    ):
        self._plot_loss(state)
        if len(self.results["return"]) == 0 or state.global_step % 10 == 0:
            self._run_eval_and_plot(model, state)

    def _run_eval_and_plot(self, agent: Agent, state: TrainerState, final=False):
        log(
            f"Running {'FINAL' if final else ''} evaluation (samples: {self.samples_processed}, epoch: {state.epoch}, step: {state.global_step})",
            with_tqdm=False,
        )

        if final:
            self.samples_processed = self.collator.samples_processed

        self._evaluate_agent(agent)
        self._plot_results()

        if not final:
            self.samples_processed = self.collator.samples_processed

    def _create_env(self, atari=False):
        env_name = self.user_args["env_name"]
        if atari:
            game = env_name.split("-")[1]
            _env = AtariEnv(self.device, game, self.user_args["seed"])
            return AtariVisualiser(
                _env,
                self.output_dir,
                filename=f"tmp",
                seed=self.user_args["seed"],
            )

        _env = gym.make(self.user_args["env_name"], render_mode="rgb_array")
        return Visualiser(_env, self.output_dir, filename=f"tmp", seed=self.user_args["seed"])

    def _record_return(self, env, ret, ep_length, model_name):
        self.results["samples"].append(self.samples_processed)
        self.results["return"].append(ret)
        self.results["episode length"].append(ep_length)
        self.results["model"].append(model_name)

        if self.user_args["record_video"]:
            env.release()

        if ret > self.best_return:
            self.best_return = ret
            log(f"New best return: {ret}", with_tqdm=False)
            if self.user_args["record_video"]:
                env.save_as_best()

    def _evaluate_agent(self, agent: Agent):
        atari = self.user_args["env_name"].startswith("atari")
        run_agent = self._run_agent_on_atari_env if atari else self._run_agent_on_env

        # for each repeat, record the episode return (and optionally render a video of the episode)
        for _ in range(self.num_repeats):
            # random baseline
            env = self._create_env(atari=atari)
            ret, ep_length = run_agent(self.random_agent, env, 0)
            self.record_return(env, ret, ep_length, "random")

            # agent under evaluation
            for tr in self.target_returns:
                env = self._create_env(atari=atari)
                ret, ep_length = run_agent(agent, env, tr)
                self.record_return(env, ret, ep_length, f"DT ({tr})")

        # convert results to dataframe
        df = pd.DataFrame(self.results)

        # log the average episode return for the current eval
        log(
            f"Average episode return: {df[df['samples'] == self.samples_processed]['return'].mean()}",
            with_tqdm=False,
        )

        # save the results to disk
        df.to_pickle(os.path.join(self.output_dir, "returns.pkl"))

    def _run_agent_on_env(self, agent: Agent, env: Visualiser):
        max_ep_len = env.max_steps if hasattr(env, "max_steps") else 1000

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

    def _run_agent_on_atari_env(
        self, agent: Agent, env: AtariVisualiser, target_return: int, stack_size=4
    ):
        def get_state(frames):
            return frames.reshape(1, self.collator.state_dim).to(
                device=self.device, dtype=torch.float32
            )

        max_ep_len = env.max_steps if hasattr(env, "max_steps") else 1000

        state_mean = torch.from_numpy(self.collator.state_mean.astype(np.float32)).to(
            device=self.device
        )
        state_std = torch.from_numpy(self.collator.state_std.astype(np.float32)).to(
            device=self.device
        )

        (obs,) = env.reset(seed=self.user_args["seed"])
        states = get_state(obs)

        print(f"states.shape: {states.shape}")

        actions = torch.zeros(
            (0, self.collator.act_dim), device=self.device, dtype=torch.float32
        )
        rewards = torch.zeros(0, device=self.device, dtype=torch.float32)
        returns_to_go = torch.tensor(
            target_return, device=self.device, dtype=torch.float32
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

            obs, r, done, _, _ = env.step(np.argmax(a))

            cur_state = get_state(obs)
            states = torch.cat([states, cur_state], dim=0)

            rewards[-1] = r
            pred_return = returns_to_go[0, -1] - r
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

        return discounted_cumsum(rewards.detach().cpu().numpy(), self.gamma)[0], t

    def _plot_loss(self, state: TrainerState):
        fig, ax = plt.subplots()
        losses = [x["loss"] for x in state.log_history]
        sns.lineplot(x=range(len(losses)), y=losses, ax=ax)
        fig.savefig(os.path.join(self.output_dir, "loss.png"))
        plt.close(fig)

    def _plot_results(self):
        fig, ax = plt.subplots()
        df = pd.DataFrame(self.results)
        sns.lineplot(x="samples", y="return", hue="model", data=df, ax=ax)
        fig.savefig(os.path.join(self.output_dir, "returns.png"))
        plt.close(fig)

        fig, ax = plt.subplots()
        sns.lineplot(x="samples", y="ep_length", hue="model", data=df, ax=ax)
        fig.savefig(os.path.join(self.output_dir, "ep_length.png"))
        plt.close(fig)
