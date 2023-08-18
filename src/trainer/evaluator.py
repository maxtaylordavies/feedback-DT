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
from src.utils.utils import log, get_minigrid_obs, normalise
from .atari_env import AtariEnv
from .visualiser import Visualiser, AtariVisualiser

sns.set_theme()


class Evaluator(TrainerCallback):
    def __init__(
        self,
        user_args,
        collator,
        sample_interval=20000,
        target_return=90,
        num_repeats=5,
        gamma=0.99,
    ) -> None:
        super().__init__()
        self.user_args = user_args
        self.collator = collator
        self.sample_interval = sample_interval
        self.samples_processed = 0
        self.target_return = target_return
        self.best_returns = {"random": -np.inf, "DT": -np.inf}
        self.best_lengths = {"random": 0, "DT": 0}
        self.num_repeats = num_repeats
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seeds = None

        # create the output directory if it doesn't exist
        self.output_dir = os.path.join(self.user_args["output"], self.user_args["run_name"])
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # initialise a dict to store the evaluation results
        self._init_results()

        # # create default missing feedback embeddings
        # self.feedback_embeddings = self.collator.embed_feedback(
        #     np.array(["No feedback available."] * user_args["context_length"])
        # ).to(self.device)

        # # create default missing mission embeddings
        # self.mission_embeddings = self.collator.embed_missions(
        #     np.array(["No mission available."] * user_args["context_length"])
        # ).to(self.device)
        self.feedback_embeddings = (
            torch.from_numpy(np.random.rand(1, 64, 128)).to(self.device).float()
        )
        self.mission_embeddings = (
            torch.from_numpy(np.random.rand(1, 64, 128)).to(self.device).float()
        )

        # create a random agent to evaluate against
        self.random_agent = RandomAgent(self.collator.act_dim)

    def _init_results(self):
        self.results = {
            "model": [],  # "random" or "DT"
            "samples": [],  # number of training samples processed by model
            "return": [],  # episode return
            "episode length": [],  # episode length
            "success": [],  # whether the episode was successful (bool)
            "seed": [],  # seed used for the episode
            "ood": [],  # whether the episode was out-of-distribution (bool)
        }

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Agent,
        **kwargs,
    ):
        self._plot_loss(state)

        # if this is the first step or we've reached the sample interval, run eval + update plots
        sample_diff = self.collator.samples_processed - self.samples_processed
        if self.samples_processed == 0 or sample_diff >= self.sample_interval:
            self._run_eval_and_plot(model, state, eval_type="efficiency")
            self.samples_processed = self.collator.samples_processed

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Agent,
        **kwargs,
    ):
        self._plot_loss(state)
        self._run_eval_and_plot(model, state, eval_type="generalisation")

    def _run_eval_and_plot(self, agent: Agent, state: TrainerState, eval_type: str):
        if eval_type == "efficiency":
            ood = False
        elif eval_type == "generalisation":
            ood = True
        else:
            raise ValueError(f"unsupported eval_type: {eval_type}")

        log(
            f"evaluating {eval_type} (samples: {self.collator.samples_processed}, epoch: {state.epoch}, step: {state.global_step})",
            with_tqdm=True,
        )

        # sample set of seeds for evaluation
        self._load_seed_dict(self.collator.dataset.configs[0])
        seeds = self._sample_seeds(n=self.num_repeats, ood=ood)

        # run evaluations using both the agent being trained and a random agent (for baseline comparison)
        for a, name in zip([self.random_agent, agent], ["random", "DT"]):
            self._evaluate_agent_performance(a, name, seeds, ood=ood)
            # self._evaluate_agent_predictions(a, name)

        self._plot_results()

    def _load_seed_dict(self, config):
        self.seeds = self.collator.dataset.seed_finder.load_seeds(
            self.user_args["level"], config
        )

    def _sample_seeds(self, n=1, ood=False, ood_type=None):
        if self.seeds is None:
            raise Exception("No seeds loaded")

        all_ood_seeds = []
        types = [k for k in self.seeds if "seed" not in k]
        for t in types:
            all_ood_seeds += self.seeds[t]["test_seeds"]

        not_ood_seeds = np.setdiff1d(
            np.arange(self.seeds["last_seed_tested"] + 1), all_ood_seeds
        )

        if ood and ood_type:
            return np.random.choice(self.seeds[ood_type]["test_seeds"], size=n, replace=False)
        elif ood:
            return np.random.choice(all_ood_seeds, size=n, replace=False)
        else:
            return np.random.choice(not_ood_seeds, size=n, replace=False)

    def _create_env(self, seed, atari=False):
        env_name = self.user_args["level"]

        if atari:
            game = env_name.split(":")[1]
            _env = AtariEnv(self.device, game, seed)
            return AtariVisualiser(
                _env,
                self.output_dir,
                filename=f"tmp",
                seed=seed,
            )

        _env = gym.make(self.user_args["env_name"], render_mode="rgb_array")
        return Visualiser(_env, self.output_dir, filename=f"tmp", seed=seed)

    def _record_result(self, env, model_name, ret, ep_length, success, seed, ood):
        self.results["samples"].append(self.collator.samples_processed)
        self.results["model"].append(model_name)
        self.results["return"].append(ret)
        self.results["episode length"].append(ep_length)
        self.results["success"].append(success)
        self.results["seed"].append(seed)
        self.results["ood"].append(ood)

        if self.user_args["record_video"]:
            env.release()

        if self.samples_processed == 0:
            env.save_as(f"first_{model_name}")

        if ret > self.best_returns[model_name]:
            self.best_returns[model_name] = ret
            log(f"new best return for {model_name} agent: {ret}", with_tqdm=True)
            if self.user_args["record_video"]:
                env.save_as(f"best_{model_name}")

        if ep_length > self.best_lengths[model_name]:
            self.best_lengths[model_name] = ep_length
            log(f"new best length for {model_name} agent: {ep_length}", with_tqdm=True)
            if self.user_args["record_video"]:
                env.save_as(f"longest_{model_name}")

    def _evaluate_agent_performance(self, agent: Agent, agent_name: str, seeds, ood=False):
        atari = self.user_args["env_name"].startswith("atari")
        run_agent = self._run_agent_on_atari_env if atari else self._run_agent_on_minigrid_env

        # for each repeat, record the episode return (and optionally render a video of the episode)
        for seed in seeds:
            env = self._create_env(seed, atari=atari)
            ret, ep_length, success = run_agent(agent, env, self.target_return)
            self._record_result(env, agent_name, ret, ep_length, success, seed, ood)

        # convert results to dataframe
        df = pd.DataFrame(self.results)

        # log the average episode return for the current eval
        log(
            f"average return ({agent_name} agent): {df[df['samples'] == self.collator.samples_processed]['return'].mean()}",
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

            self.results["samples"].append(self.collator.samples_processed)
            self.results["model"].append(agent_name)
            self.results["eval acc"].append(accs[rep])

        log(
            f"eval acc: {np.mean(accs)} (after {self.collator.samples_processed} samples)",
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

        max_ep_len = env.max_steps if hasattr(env, "max_steps") else 64
        obs, _ = env.reset(seed=self.user_args["seed"])

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
                    mission_embeddings=self.mission_embeddings[:, : t + 1, :],
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    returns_to_go=returns_to_go,
                    timesteps=timesteps,
                    feedback_embeddings=self.feedback_embeddings[:, : t + 1, :],
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

        success = done and reward > 0
        return np.sum(rewards.detach().cpu().numpy()), t, success

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
        formats = ["png", "svg"]

        # split into in-distribution and out-of-distribution for efficiency and generalisation plots
        df = pd.DataFrame(self.results)
        eff_df = df[df["ood"] == False]
        gen_df = df[df["ood"] == True]

        metrics = set(self.results.keys()).difference({"samples", "model", "seed", "ood"})
        for m in metrics:
            # for success, we want the percentage success rate, which we
            # can get by taking the mean of the success column multiplied by 100
            if m == "success":
                df[m] = df[m] * 100

            # first, do line plot against samples (for sample efficiency)
            fig, ax = plt.subplots()
            sns.lineplot(x="samples", y=m, hue="model", data=eff_df, ax=ax)
            for fmt in formats:
                fig.savefig(os.path.join(self.output_dir, f"eff_{m.replace(' ', '_')}.{fmt}"))
            plt.close(fig)

            # then, do bar plot (for generalisation) (if we have data yet)
            if len(gen_df) > 0:
                fig, ax = plt.subplots()
                sns.barplot(x="model", y=m, data=gen_df, ax=ax)
                for fmt in formats:
                    fig.savefig(
                        os.path.join(self.output_dir, f"gen_{m.replace(' ', '_')}.{fmt}")
                    )
                plt.close(fig)
