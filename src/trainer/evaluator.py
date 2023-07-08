import os

import gymnasium as gym
import torch
import numpy as np
import pandas as pd

# from minigrid.wrappers import FullyObsWrapper
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
import matplotlib.pyplot as plt
import seaborn as sns

from src.agent import Agent, AgentInput, RandomAgent
from src.utils.utils import discounted_cumsum, log, get_minigrid_obs, normalise
from .atari_env import AtariEnv
from .visualiser import Visualiser, AtariVisualiser

sns.set_theme()


class Evaluator(TrainerCallback):
    def __init__(
        self,
        user_args,
        collator,
        sample_interval=1000,
        target_returns=[90],
        num_repeats=3,
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

        # initialise a dict to store the evaluation results
        self.results = {
            "model": [],
            "samples": [],
            "return": [],
            "episode length": [],
            "eval acc": [],
        }

        # create default missing feedback embeddings
        self.feedback_embeddings = self.collator.embed_feedback(
            np.array([["No feedback available."] * user_args["context_length"]])
        ).to(self.device)

        # create default missing mission embeddings
        self.mission_embeddings = self.collator.embed_missions(
            np.array([["No mission available."] * user_args["context_length"]])
        ).to(self.device)

        # create a random agent to evaluate against
        self.random_agent = RandomAgent(self.collator.act_dim)

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
        if len(self.results["samples"]) == 0 or sample_diff >= self.sample_interval:
            self._run_eval_and_plot(model, state)

    def _run_eval_and_plot(self, agent: Agent, state: TrainerState, final=False):
        log(
            f"Running {'FINAL' if final else ''} evaluation (samples: {self.samples_processed}, epoch: {state.epoch}, step: {state.global_step})",
            with_tqdm=True,
        )

        # run evaluations using both the agent being trained and a random agent (for baseline comparison)
        for a, name in zip([agent, self.random_agent], ["DT", "random"]):
            self._evaluate_agent_performance(a, name)
            self._evaluate_agent_predictions(a, name)

        self._plot_results()
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

    def _record_return(self, env, ret, ep_length, model_name):
        self.results["samples"].append(self.samples_processed)
        self.results["return"].append(ret)
        self.results["episode length"].append(ep_length)
        self.results["model"].append(model_name)

        if self.user_args["record_video"]:
            env.release()

        if self.samples_processed == 0:
            env.save_as("first")

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

    def _evaluate_agent_performance(self, agent: Agent, agent_name: str):
        atari = self.user_args["env_name"].startswith("atari")
        run_agent = self._run_agent_on_atari_env if atari else self._run_agent_on_minigrid_env

        # for each repeat, record the episode return (and optionally render a video of the episode)
        for _ in range(self.num_repeats):
            for tr in self.target_returns:
                env = self._create_env(atari=atari)
                ret, ep_length = run_agent(agent, env, tr)
                self._record_return(env, ret, ep_length, f"{agent_name} ({tr})")

        # convert results to dataframe
        df = pd.DataFrame(self.results)

        # log the average episode return for the current eval
        log(
            f"Average episode return: {df[df['samples'] == self.samples_processed]['return'].mean()}",
            with_tqdm=True,
        )

        # save the results to disk
        df.to_pickle(os.path.join(self.output_dir, "results.pkl"))

    def _evaluate_agent_predictions(self, agent, agent_name, repeats=10, num_steps=100):
        accs = np.zeros(repeats)

        for rep in range(repeats):
            batch = self.collator._sample_batch(1, random_start=True, full=True, train=False)
            for k in batch:
                batch[k] = batch[k].to(self.device)

            ns = min(num_steps, batch["states"].shape[1] - 30)
            for i in range(ns):
                input = AgentInput(
                    states=batch["states"][:, i : i + 30],
                    actions=batch["actions"][:, i : i + 30],
                    rewards=batch["rewards"][:, i : i + 30],
                    returns_to_go=batch["returns_to_go"][:, i : i + 30],
                    timesteps=batch["timesteps"][:, i : i + 30],
                    feedback_embeddings=batch["feedback_embeddings"][:, i : i + 30],
                    attention_mask=batch["attention_mask"][:, i : i + 30],
                )

                action = agent.get_action(input)
                got = np.argmax(action.detach().cpu().numpy())
                target = np.argmax(input.actions[0, -1].detach().cpu().numpy())

                accs[rep] += int(got == target) / ns

            self.results["samples"].append(self.samples_processed)
            self.results["model"].append(agent_name)
            self.results["eval acc"].append(accs[rep])

        log(
            f"eval acc: {np.mean(accs)} (after {self.samples_processed} samples)",
            with_tqdm=True,
        )

    def _run_agent_on_minigrid_env(self, agent: Agent, env: Visualiser, target_return: float):
        def get_state(partial_obs):
            obs = get_minigrid_obs(
                env.get_env(),
                partial_obs,
                self.user_args["fully_obs"],
                self.user_args["rgb_obs"],
            )
            return (
                torch.from_numpy(normalise(obs["image"]))
                .reshape(1, self.collator.state_dim)
                .to(device=self.device, dtype=torch.float32)
            )

        max_ep_len = env.max_steps if hasattr(env, "max_steps") else 1000

        obs, _ = env.reset(seed=self.user_args["seed"])
        # fully_obs_env = FullyObsWrapper(env)
        # obs = fully_obs_env.observation(tmp[0])

        states = get_state(obs)
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
                [actions, torch.zeros((1, self.collator.act_dim), device=self.device)],
                dim=0,
            )
            rewards = torch.cat([rewards, torch.zeros(1, device=self.device)])

            actions[-1] = agent.get_action(
                AgentInput(
                    mission_embeddings=self.mission_embeddings,
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
            a = actions[-1].detach().cpu().numpy()

            obs, reward, done, _, _ = env.step(np.argmax(a))
            # obs = fully_obs_env.observation(obs)
            cur_state = get_state(obs)
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
        self, agent: Agent, env: AtariVisualiser, target_return: float, stack_size=4
    ):
        def get_state(frames):
            frames = frames.permute(1, 2, 0)
            return normalise(frames.reshape(1, self.collator.state_dim)).to(
                device=self.device, dtype=torch.float32
            )

        max_ep_len = env.max_steps if hasattr(env, "max_steps") else 5000

        obs = env.reset(seed=self.user_args["seed"])
        init_s = get_state(obs).detach().cpu().numpy()

        _start_idx = 655
        NUM_INITIAL_ACTIONS = 10

        # ref_states = normalise(self.collator.observations[_start_idx : _start_idx + NUM_INITIAL_ACTIONS])

        # states = get_state(obs)
        states = torch.zeros(0, device=self.device, dtype=torch.float32)
        actions = torch.tensor(
            self.collator.actions[_start_idx : _start_idx + NUM_INITIAL_ACTIONS],
            device=self.device,
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
            s = get_state(obs)

            states = torch.cat([states, s], dim=0)
            rewards = torch.cat([rewards, torch.tensor(r, device=self.device).reshape(1)])

            if i == 0:
                returns_to_go = torch.tensor(
                    target_return, device=self.device, dtype=torch.float32
                ).reshape(1, 1)
            else:
                returns_to_go = torch.cat(
                    [returns_to_go, (returns_to_go[0, -1] - r).reshape(1, 1)], dim=1
                )

        for t in range(NUM_INITIAL_ACTIONS, max_ep_len):
            # get action from agent
            a = agent.get_action(
                AgentInput(
                    mission_embeddings=self.mission_embeddings,
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
                break

        # return discounted_cumsum(rewards.detach().cpu().numpy(), self.gamma)[0], t
        return np.sum(rewards.detach().cpu().numpy()), t

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
        fig.savefig(os.path.join(self.output_dir, "returns.png"))
        plt.close(fig)

        fig, ax = plt.subplots()
        sns.lineplot(x="samples", y="episode length", hue="model", data=df, ax=ax)
        fig.savefig(os.path.join(self.output_dir, "ep-length.png"))
        plt.close(fig)

        fig, ax = plt.subplots()
        sns.lineplot(x="samples", y="eval acc", hue="model", data=df, ax=ax)
        fig.savefig(os.path.join(self.output_dir, "eval-acc.png"))
        plt.close(fig)
