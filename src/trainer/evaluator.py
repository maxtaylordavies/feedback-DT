import os
import warnings

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from jsonc_parser.parser import JsoncParser
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from src.agent import Agent, AgentInput, RandomAgent
from src.utils.utils import log, get_minigrid_obs, normalise
from src.env.recorder_env import RecorderEnv
from src.collator import CurriculumCollator, RoundRobinCollator
from src.dataset.custom_feedback_verifier import TaskFeedback

warnings.filterwarnings("ignore")

sns.set_theme()


class Evaluator(TrainerCallback):
    def __init__(
        self,
        user_args,
        collator,
    ) -> None:
        super().__init__()
        self.user_args = user_args
        self.collator = collator
        self.sample_interval = self.user_args["sample_interval"]
        self.target_return = self.user_args["target_return"]
        self.num_repeats = self.user_args["num_repeats"]
        self.samples_processed = 0
        self.best_returns = {"random": -np.inf, "DT": -np.inf}
        self.best_lengths = {"random": np.inf, "DT": np.inf}
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
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
            torch.from_numpy(np.random.rand(1, self.user_args["context_length"], 128))
            .float()
            .to(self.device)
        )
        self.mission_embeddings = (
            torch.from_numpy(np.random.rand(1, self.user_args["context_length"], 128))
            .float()
            .to(self.device)
        )

        # create a random agent to evaluate against
        self.random_agent = RandomAgent(self.collator.act_dim)
        self.current_epoch = 0

    def _init_results(self):
        self.results = {
            "model": [],  # "random" or "DT"
            "samples": [],  # number of training samples processed by model
            "level": [],  # level name (for MT training)
            "config": [],  # config used for the episode
            "seed": [],  # seed used for the episode
            "ood_type": [],  # type of out-of-distribution episode (if applicable, else empty string)
            "return": [],  # episode return
            "episode length": [],  # episode length
            "success": [],  # whether the episode was successful (bool)
            "gc_success": [],  # goal condition success rate (float)
            "pw_success": [],  # path-weighted success rate (float)
        }

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Agent,
        **kwargs,
    ):
        log("on_train_begin called", with_tqdm=True)

        # run initial eval (before any training steps)
        self._run_eval_and_plot(model, state, eval_type="efficiency")

        return super().on_train_begin(args, state, control, **kwargs)

    def on_epoch_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        log("on_epoch_begin called", with_tqdm=True)
        return super().on_epoch_begin(args, state, control, **kwargs)

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Agent,
        **kwargs,
    ):
        log("on_step_begin called", with_tqdm=True)

        self._plot_loss(state)

        # if this is the first step or we've reached the sample interval, run eval + update plots
        sample_diff = self.collator.samples_processed - self.samples_processed
        if sample_diff >= self.sample_interval:
            self._run_eval_and_plot(model, state, eval_type="efficiency")
            self.samples_processed = self.collator.samples_processed

        previous_epoch = self.current_epoch
        self.current_epoch = state.epoch
        if previous_epoch != self.current_epoch:
            self.collator.update_epoch()

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
        if eval_type not in ["efficiency", "generalisation"]:
            raise Exception(f"Unknown eval type: {eval_type}")

        log(
            f"evaluating {eval_type} (samples: {self.collator.samples_processed}, epoch: {state.epoch}, step: {state.global_step})",
            with_tqdm=True,
        )

        if isinstance(self.collator, CurriculumCollator) or isinstance(
            self.collator, RoundRobinCollator
        ):
            for dataset in self.collator.datasets:
                self._evaluate_by_config(dataset, agent, eval_type)
        else:
            self._evaluate_by_config(self.collator.dataset, agent, eval_type)

        self._plot_results()

    def _evaluate_by_config(self, dataset, agent, eval_type):
        total_repeats = (
            self.num_repeats if eval_type == "generalisation" else self.num_repeats // 4
        )
        per_config_repeats = total_repeats // len(dataset.configs) + (
            total_repeats % len(dataset.configs) > 0
        )

        n_seeds_sampled = 0
        for config in dataset.configs:
            # sample some seeds for the current config
            if n_seeds_sampled + per_config_repeats <= total_repeats:
                n_seeds_to_sample = per_config_repeats
            else:
                n_seeds_to_sample = total_repeats - n_seeds_sampled
            self._load_seed_dict(dataset, config)
            seeds = (
                self._sample_validation_seeds(n=n_seeds_to_sample)
                if eval_type == "efficiency"
                else self._sample_test_seeds(n=n_seeds_to_sample)
            )

            # run evaluations using both the agent being trained and a random agent (for baseline comparison)
            for a, name in zip([self.random_agent, agent], ["random", "DT"]):
                self._evaluate_agent_performance(a, name, dataset, config, seeds)

            n_seeds_sampled += n_seeds_to_sample

    def _load_seed_dict(self, dataset, config):
        self.seeds = dataset.seed_finder.load_seeds(dataset.level, config)

    def _sample_validation_seeds(self, n=1):
        if self.seeds is None:
            raise Exception("No seeds loaded")

        # to conform to the same interface as _sample_test_seeds, i.e. {ood_type: [seeds]}
        return {"": np.random.choice(self.seeds["validation_seeds"], size=n, replace=False)}

    def _sample_test_seeds(self, n=1):
        if self.seeds is None:
            raise Exception("No seeds loaded")

        seeds = {}
        types = [k for k in self.seeds if "seed" not in k and self.seeds[k]["test_seeds"]]
        per_type_repeats = n // len(types) + (n % len(types) > 0)

        seeds_sampled = 0
        for t in types:
            if seeds_sampled + per_type_repeats <= n:
                n_seeds_to_sample = per_type_repeats
            else:
                n_seeds_to_sample = n - seeds_sampled
            try:
                seeds[t] = np.random.choice(
                    self.seeds[t]["test_seeds"], size=n_seeds_to_sample, replace=False
                )
            except ValueError:
                seeds[t] = np.random.choice(
                    self.seeds[t]["test_seeds"], size=n_seeds_to_sample, replace=True
                )
            seeds_sampled += n_seeds_to_sample
        return seeds

    def _create_env(self, config, seed):
        # if "atari" in config:
        #     game = config.split(":")[1]
        #     _env = AtariEnv(self.device, game, seed)
        #     return AtariRecorderEnv(
        #         _env,
        #         self.user_args["feedback_mode"],
        #         self.output_dir,
        #         filename=f"tmp",
        #     )

        _env = gym.make(config, render_mode="rgb_array")
        env = RecorderEnv(
            _env, self.user_args["feedback_mode"], self.output_dir, filename=f"tmp"
        )
        env.reset(seed=seed)
        return env

    def _record_result(
        self,
        env,
        dataset,
        config,
        seed,
        ood_type,
        model_name,
        ret,
        ep_length,
        success,
        gc_success,
    ):
        self.results["model"].append(model_name)
        self.results["samples"].append(self.collator.samples_processed)
        self.results["level"].append(dataset.level)
        self.results["config"].append(config)
        self.results["seed"].append(seed)
        self.results["ood_type"].append(ood_type)
        self.results["return"].append(ret)
        self.results["episode length"].append(ep_length)
        self.results["success"].append(success)
        self.results["gc_success"].append(gc_success)
        self.results["pw_success"].append(
            self._get_pw_success(success, ep_length, dataset.level)
        )

        if self.user_args["record_video"]:
            env.release()

        if self.samples_processed == 0:
            env.save_as(f"first_{model_name}")

        if ret > self.best_returns[model_name]:
            self.best_returns[model_name] = ret
            # log(f"new best return for {model_name} agent: {ret}", with_tqdm=True)
            if self.user_args["record_video"]:
                env.save_as(f"best_{model_name}")

        if ep_length < self.best_lengths[model_name]:
            self.best_lengths[model_name] = ep_length
            # log(f"new best length for {model_name} agent: {ep_length}", with_tqdm=True)
            if self.user_args["record_video"]:
                env.save_as(f"longest_{model_name}")

    def _evaluate_agent_performance(
        self, agent: Agent, agent_name: str, dataset, config: str, seeds: dict
    ):
        run_agent = self._run_agent_on_minigrid_env

        # for each repeat, run agent and record metrics (and optionally render a video of the episode)
        for ood_type, seeds in seeds.items():  # ood_type is "" if not ood
            for seed in seeds:
                seed = int(seed)
                env = self._create_env(config, seed)
                ret, ep_length, success, gc_success = run_agent(
                    agent, env, seed, self.target_return
                )
                self._record_result(
                    env,
                    dataset,
                    config,
                    seed,
                    ood_type,
                    agent_name,
                    ret,
                    ep_length,
                    success,
                    gc_success,
                )

        # convert results to dataframe
        df = pd.DataFrame(self.results)

        # log the average episode return for the current eval
        # log(
        #     f"average return ({agent_name} agent) on level {dataset.level}: {df[df['samples'] == self.collator.samples_processed]['return'].mean()}",
        #     with_tqdm=True,
        # )

        # save the results to disk
        df.to_pickle(os.path.join(self.output_dir, "results.pkl"))

    def _get_demo_mean(self, level):
        metadata_path = os.getenv("ENV_METADATA_PATH", "env_metadata.jsonc")
        metadata = JsoncParser.parse_file(metadata_path)["levels"]
        for level_group, levels in metadata.items():
            if level in levels:
                return round(metadata[level_group][level]["demo_mean_n_steps"])

    def _get_pw_success(self, success, episode_length, level):
        demo_length = self._get_demo_mean(level)
        return success * (demo_length / max(episode_length, demo_length))

    def _run_agent_on_minigrid_env(
        self, agent: Agent, env: RecorderEnv, seed: int, target_return: float
    ):
        def get_state(partial_obs):
            obs = get_minigrid_obs(
                env,
                partial_obs,
                self.user_args["fully_obs"],
                self.user_args["rgb_obs"],
            )
            return (
                torch.from_numpy(normalise(obs["image"]))
                .reshape(1, self.collator.state_dim)
                .to(device=self.device, dtype=torch.float32)
            )

        max_ep_len = env.max_steps
        obs, _ = env.reset(seed=seed)

        states = get_state(obs)
        actions = torch.zeros(
            (0, self.collator.act_dim), device=self.device, dtype=torch.float32
        )
        rewards = torch.zeros(0, device=self.device, dtype=torch.float32)
        returns_to_go = torch.tensor(
            target_return, device=self.device, dtype=torch.float32
        ).reshape(1, 1)
        timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)

        task_feedback_verifier = TaskFeedback(env)
        goal_conditions = len(task_feedback_verifier.subtasks)
        goal_conditions_met = 0

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

            goal_conditions_met += (
                task_feedback_verifier.verify_feedback(env, np.argmax(a))
                != "No feedback available."
            )

            if done:
                break

        success = done and reward > 0
        gc_sr = goal_conditions_met / goal_conditions
        return np.sum(rewards.detach().cpu().numpy()), t, success, gc_sr

    # def _run_agent_on_atari_env(
    #     self, agent: Agent, env: AtariRecorderEnv, target_return: float, stack_size=4
    # ):
    #     def get_state(frames):
    #         frames = frames.permute(1, 2, 0)
    #         return normalise(frames.reshape(1, self.collator.state_dim)).to(
    #             device=self.device, dtype=torch.float32
    #         )

    #     max_ep_len = env.max_steps if hasattr(env, "max_steps") else 5000

    #     obs = env.reset(seed=self.user_args["seed"])
    #     init_s = get_state(obs).detach().cpu().numpy()

    #     _start_idx = 655
    #     NUM_INITIAL_ACTIONS = 10

    #     # ref_states = normalise(self.collator.observations[_start_idx : _start_idx + NUM_INITIAL_ACTIONS])

    #     # states = get_state(obs)
    #     states = torch.zeros(0, device=self.device, dtype=torch.float32)
    #     actions = torch.tensor(
    #         self.collator.actions[_start_idx : _start_idx + NUM_INITIAL_ACTIONS],
    #         device=self.device,
    #     )
    #     rewards = torch.zeros(0, device=self.device, dtype=torch.float32)
    #     returns_to_go = torch.zeros(0, device=self.device, dtype=torch.float32)
    #     timesteps = torch.zeros(0, device=self.device, dtype=torch.long)

    #     # execute set of initial actions to get going
    #     a_idxs = []
    #     for i, a in enumerate(actions):
    #         a_idx = np.argmax(a.detach().cpu().numpy())
    #         a_idxs.append(a_idx)

    #         obs, r, _ = env.step(a_idx)
    #         s = get_state(obs)

    #         states = torch.cat([states, s], dim=0)
    #         rewards = torch.cat(
    #             [rewards, torch.tensor(r, device=self.device).reshape(1)]
    #         )

    #         if i == 0:
    #             returns_to_go = torch.tensor(
    #                 target_return, device=self.device, dtype=torch.float32
    #             ).reshape(1, 1)
    #         else:
    #             returns_to_go = torch.cat(
    #                 [returns_to_go, (returns_to_go[0, -1] - r).reshape(1, 1)], dim=1
    #             )

    #     for t in range(NUM_INITIAL_ACTIONS, max_ep_len):
    #         # get action from agent
    #         a = agent.get_action(
    #             AgentInput(
    #                 mission_embeddings=self.mission_embeddings,
    #                 states=states,
    #                 actions=actions,
    #                 rewards=rewards,
    #                 returns_to_go=returns_to_go,
    #                 timesteps=timesteps,
    #                 feedback_embeddings=self.feedback_embeddings,
    #                 attention_mask=None,
    #             ),
    #             context=self.user_args["context_length"],
    #             one_hot=True,
    #         )

    #         # take action, get next state and reward
    #         a_idx = np.argmax(a.detach().cpu().numpy())
    #         a_idxs.append(a_idx)
    #         obs, r, done = env.step(a_idx)
    #         s = get_state(obs)

    #         # update state, reward, return, timestep tensors
    #         states = torch.cat([states, s], dim=0)
    #         actions = torch.cat([actions, a.reshape((1, 4))], dim=0)
    #         rewards = torch.cat(
    #             [rewards, torch.tensor(r, device=self.device).reshape(1)]
    #         )
    #         returns_to_go = torch.cat(
    #             [returns_to_go, (returns_to_go[0, -1] - r).reshape(1, 1)], dim=1
    #         )

    #         timesteps = torch.cat(
    #             [
    #                 timesteps,
    #                 torch.ones((1, 1), device=self.device, dtype=torch.long) * t,
    #             ],
    #             dim=1,
    #         )

    #         if done:
    #             break

    #     # return discounted_cumsum(rewards.detach().cpu().numpy(), self.gamma)[0], t
    #     return np.sum(rewards.detach().cpu().numpy()), t

    def _plot_loss(self, state: TrainerState):
        fig, ax = plt.subplots()
        losses = [x["loss"] for x in state.log_history[:-1]]
        sns.lineplot(x=range(len(losses)), y=losses, ax=ax)
        fig.savefig(os.path.join(self.output_dir, "loss.png"))
        plt.close(fig)

    def _plot_results(self):
        formats = ["png", "svg"]

        # split into in-distribution and out-of-distribution for efficiency and generalisation plots
        df = pd.DataFrame(self.results)
        eff_df = df[df["ood_type"] == ""]
        gen_df = df[df["ood_type"] != ""]

        metrics = set(self.results.keys()).difference(
            {"samples", "model", "level", "config", "seed", "ood_type"}
        )
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
                sns.barplot(x="model", y=m, hue="ood_type", data=gen_df, ax=ax)
                for fmt in formats:
                    fig.savefig(
                        os.path.join(self.output_dir, f"gen_{m.replace(' ', '_')}.{fmt}")
                    )
                plt.close(fig)
