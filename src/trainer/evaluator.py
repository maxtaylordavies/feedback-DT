import os
import shutil

import cv2
import gymnasium as gym
import torch
import numpy as np
import pandas as pd

# from minigrid.wrappers import FullyObsWrapper
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


class Evaluator(TrainerCallback):
    def __init__(
        self,
        user_args,
        collator,
        sample_interval=1000,
        target_returns=[90],
        num_repeats=5,
        gamma=0.99,
    ) -> None:
        super().__init__()
        self.user_args = user_args
        self.collator = collator
        self.sample_interval = sample_interval
        self.samples_processed = 0
        self.target_returns = target_returns
        self.best_return = -np.inf
        self.best_random_return = -np.inf
        self.best_length = 0
        self.num_repeats = num_repeats
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

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

        sample_diff = self.collator.samples_processed - self.samples_processed
        if len(self.results["return"]) == 0 or sample_diff >= self.sample_interval:
            self._run_eval_and_plot(model, state)

    def _run_eval_and_plot(self, agent: Agent, state: TrainerState, final=False):
        log(
            f"Running {'FINAL' if final else ''} evaluation (samples: {self.samples_processed}, epoch: {state.epoch}, step: {state.global_step})",
            with_tqdm=True,
        )

        if final:
            self.samples_processed = self.collator.samples_processed

        self._evaluate_agent(agent, final=final)
        self._plot_results()

        if not final:
            self.samples_processed = self.collator.samples_processed

    def _create_env(self, atari=False):
        env_name = self.user_args["env_name"]
        if atari:
            game = env_name.split(":")[1]
            _env = AtariEnv(self.device, game, self.user_args["seed"])
            return AtariVisualiser(
                _env,
                self.output_dir,
                filename=f"tmp",
                seed=self.user_args["seed"],
            )

        _env = gym.make(self.user_args["env_name"], render_mode="rgb_array")
        return Visualiser(_env, self.output_dir, filename=f"tmp", seed=self.user_args["seed"])

    def _record_return(self, env, ret, ep_length, model_name, final=False):
        self.results["samples"].append(self.samples_processed)
        self.results["return"].append(ret)
        self.results["episode length"].append(ep_length)
        self.results["model"].append(model_name)

        if self.user_args["record_video"]:
            env.release()

        if self.samples_processed == 0:
            env.save_as("first")
        elif final:
            env.save_as("final")

        if model_name == "random" and ret > self.best_random_return:
            self.best_random_return = ret
            log(f"New best random return: {ret}", with_tqdm=True)
            if self.user_args["record_video"]:
                env.save_as("best_random")

        if model_name != "random" and ret > self.best_return:
            self.best_return = ret
            log(f"New best return: {ret}", with_tqdm=True)
            if self.user_args["record_video"]:
                env.save_as("best")

        if model_name != "random" and ep_length > self.best_length:
            self.best_length = ep_length
            log(f"New best length: {ep_length}", with_tqdm=True)
            if self.user_args["record_video"]:
                env.save_as("longest")

    def _evaluate_agent(self, agent: Agent, final=False):
        atari = self.user_args["env_name"].startswith("atari")
        run_agent = self._run_agent_on_atari_env if atari else self._run_agent_on_env

        # for each repeat, record the episode return (and optionally render a video of the episode)
        for _ in range(self.num_repeats):
            # random baseline
            env = self._create_env(atari=atari)
            ret, ep_length = run_agent(self.random_agent, env, 0)
            self._record_return(env, ret, ep_length, "random")

            # agent under evaluation
            for tr in self.target_returns:
                env = self._create_env(atari=atari)
                ret, ep_length = run_agent(agent, env, tr)
                self._record_return(env, ret, ep_length, f"DT ({tr})", final=final)

        # convert results to dataframe
        df = pd.DataFrame(self.results)

        # log the average episode return for the current eval
        log(
            f"Average episode return: {df[df['samples'] == self.samples_processed]['return'].mean()}",
            with_tqdm=True,
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
            return self.collator._normalise_states(
                frames.reshape(1, self.collator.state_dim)
            ).to(device=self.device, dtype=torch.float32)

        max_ep_len = env.max_steps if hasattr(env, "max_steps") else 5000

        obs = env.reset(seed=self.user_args["seed"])
        _start_idx = self.collator.episode_starts[-1]

        # states = get_state(obs)
        states = torch.zeros(0, device=self.device, dtype=torch.float32)
        actions = torch.tensor(
            self.collator.actions[_start_idx : _start_idx + 20], device=self.device
        )
        rewards = torch.zeros(0, device=self.device, dtype=torch.float32)
        returns_to_go = torch.zeros(0, device=self.device, dtype=torch.float32)
        timesteps = torch.zeros(0, device=self.device, dtype=torch.long)

        # execute set of initial actions to get going
        a_idxs = []
        for i, a in enumerate(actions):
            a_idx = np.argmax(a.detach().cpu().numpy())
            a_idxs.append(a_idx)
            obs, r, _ = env.step(a_idx)
            states = torch.cat([states, get_state(obs)], dim=0)
            rewards = torch.cat([rewards, torch.tensor(r, device=self.device).reshape(1)])

            if i == 0:
                returns_to_go = torch.tensor(
                    target_return, device=self.device, dtype=torch.float32
                ).reshape(1, 1)
            else:
                returns_to_go = torch.cat(
                    [returns_to_go, (returns_to_go[0, -1] - r).reshape(1, 1)], dim=1
                )

            timesteps = torch.cat(
                [
                    timesteps,
                    torch.ones((1, 1), device=self.device, dtype=torch.long) * (i),
                ],
                dim=1,
            )

        for t in range(20, max_ep_len):
            # get action from agent
            a = agent.get_action(
                AgentInput(
                    states=states,
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

            # take action, get next state and reward
            a_idx = np.argmax(a.detach().cpu().numpy())
            a_idxs.append(a_idx)
            obs, r, done = env.step(a_idx)
            s = get_state(obs)

            # update state, reward, return, timestep tensors
            states = torch.cat([states, s], dim=0)
            actions = torch.cat([actions, a.reshape((1, 4))], dim=0)
            rewards = torch.cat([rewards, torch.tensor(r, device=self.device).reshape(1)])
            returns_to_go = torch.cat(
                [returns_to_go, (returns_to_go[0, -1] - r).reshape(1, 1)], dim=1
            )

            timesteps = torch.cat(
                [
                    timesteps,
                    torch.ones((1, 1), device=self.device, dtype=torch.long) * t,
                ],
                dim=1,
            )

            if done:
                if target_return:
                    print(a_idxs)
                break

        return discounted_cumsum(rewards.detach().cpu().numpy(), self.gamma)[0], t

    def _plot_loss(self, state: TrainerState):
        fig, ax = plt.subplots()
        losses = [x["loss"] for x in state.log_history]
        sns.lineplot(x=range(len(losses)), y=losses, ax=ax)
        fig.savefig(os.path.join(self.output_dir, "loss.png"))
        plt.close(fig)

    def _plot_results(self):
        df = pd.DataFrame(self.results)

        fig, ax = plt.subplots()
        sns.lineplot(x="samples", y="return", hue="model", data=df, ax=ax)
        fig.savefig(os.path.join(self.output_dir, "returns-mean.png"))
        plt.close(fig)

        fig, ax = plt.subplots()
        sns.lineplot(x="samples", y="return", hue="model", estimator="max", data=df, ax=ax)
        fig.savefig(os.path.join(self.output_dir, "returns-max.png"))
        plt.close(fig)

        fig, ax = plt.subplots()
        sns.lineplot(x="samples", y="episode length", hue="model", data=df, ax=ax)
        fig.savefig(os.path.join(self.output_dir, "ep-length-mean.png"))
        plt.close(fig)

        fig, ax = plt.subplots()
        sns.lineplot(
            x="samples", y="episode length", hue="model", estimator="max", data=df, ax=ax
        )
        fig.savefig(os.path.join(self.output_dir, "ep-length-max.png"))
        plt.close(fig)
