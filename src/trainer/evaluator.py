import os
import warnings

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from jsonc_parser.parser import JsoncParser
from transformers import TrainerCallback
from transformers import TrainerControl
from transformers import TrainerState
from transformers import TrainingArguments
from transformers.trainer_callback import TrainerControl
from transformers.trainer_callback import TrainerState
from transformers.training_args import TrainingArguments

from src.agent import Agent
from src.agent import AgentInput
from src.agent import RandomAgent
from src.collator import CurriculumCollator
from src.collator import RoundRobinCollator
from src.dataset.custom_feedback_verifier import TaskFeedback
from src.env.recorder_env import RecorderEnv
from src.utils.utils import get_minigrid_obs
from src.utils.utils import log
from src.utils.utils import normalise

warnings.filterwarnings("ignore")

sns.set_theme()


class Evaluator(TrainerCallback):
    def __init__(self, user_args, collator) -> None:
        super().__init__()
        self.user_args = user_args
        self.collator = collator
        self.early_stopping_patience = self.user_args["early_stopping_patience"]
        self.early_stopping_patience_counter = 0
        self.early_stopping_threshold = self.user_args["early_stopping_threshold"]
        self.best_gc_success = -np.inf
        self.best_global_step = 0
        self.sample_interval = self.user_args["sample_interval"]
        self.target_return = self.user_args["target_return"]
        self.num_repeats = min(self.user_args["num_repeats"], self.user_args["num_train_seeds"])
        self.samples_processed = 0
        self.best_returns = {"random": -np.inf, "DT": -np.inf}
        self.best_lengths = {"random": np.inf, "DT": np.inf}
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.seeds = None
        self.train_seeds = None
        self.val_seeds = None
        # create the output directory if it doesn't exist
        self.output_dir = os.path.join(
            self.user_args["output"],
            os.path.join(self.user_args["run_name"], str(self.user_args["model_seed"])),
        )
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self._init_results()

        self.str_feedback_embeddings = (
            self.collator.embed_sentences(np.array(["No feedback available."] * self.collator.dataset.max_steps), "feedback")
            .to(self.device)
        )

        self.str_mission_embeddings = (
            self.collator.embed_sentences(np.array(["No mission available."] * self.collator.dataset.max_steps), "mission")
            .to(self.device)
        )

        self.int_feedback_embeddings = (
            torch.from_numpy(np.random.rand(1, self.collator.dataset.max_steps, 128))
            .float()
            .to(self.device)
        )

        self.int_mission_embeddings = (
            torch.from_numpy(np.random.rand(1, self.collator.dataset.max_steps, 128))
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
            "eval_type": [],  # type of evaluation (efficiency, iid_generalisation, ood_generalisation)
            "ood_type": [],  # type of out-of-distribution episode (if applicable, else empty string)
            "return": [],  # episode return
            "episode_length": [],  # episode_length
            "success": [],  # whether the episode was successful (bool)
            "gc_success": [],  # goal condition success rate (float)
            "pw_success": [],  # path-weighted success rate (float)
            "global_step": [],  # global step number
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

        # sample some validation seeds for the train config
        self._load_seed_dict(self.collator.dataset, self.collator.dataset.train_config)
        self._set_train_seeds()

        # run initial eval (before any training steps)
        self._run_eval_and_plot(model, state, eval_type="efficiency")

        return super().on_train_begin(args, state, control, **kwargs)

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
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
        self._plot_loss(state)

        # if this is the first step or we've reached the sample interval, run eval + update plots
        sample_diff = self.collator.samples_processed - self.samples_processed
        if sample_diff >= self.sample_interval:
            self._run_eval_and_plot(
                model, state, eval_type="efficiency", control=control
            )
            self.samples_processed = self.collator.samples_processed
            log(
                f"saving model checkpoint after step {state.global_step}",
                with_tqdm=True,
            )
            model.save_checkpoint(self.output_dir, state.global_step)

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
        log("Training ended - loading model checkpoint")
        model.load_checkpoint(self.output_dir, self.best_global_step)
        self._plot_loss(state)
        self._run_eval_and_plot(model, state, eval_type="iid_generalisation")
        self._run_eval_and_plot(model, state, eval_type="ood_generalisation")

    def _run_eval_and_plot(
        self,
        agent: Agent,
        state: TrainerState,
        eval_type: str,
        control: TrainerControl = None,
    ):
        if eval_type not in ["efficiency", "iid_generalisation", "ood_generalisation"]:
            raise Exception(f"Unknown eval type: {eval_type}")

        log(
            f"evaluating {eval_type} (samples: {self.collator.samples_processed}, epoch: {state.epoch}, step: {state.global_step})",
            with_tqdm=True,
        )
        if eval_type == "efficiency":
            self._evaluate_efficiency(self.collator.dataset, agent, state, control)
        elif eval_type == "iid_generalisation":
            self._evaluate_iid_generalisation(self.collator.dataset, agent, state)
        else:
            self._evaluate_ood_generalisation(self.collator.dataset, agent, state)

        self._plot_results()

    def _evaluate_efficiency(
        self, dataset, agent, state: TrainerState, control: TrainerControl
    ):
        # run evaluations using both the agent being trained and a random agent (for baseline comparison)
        for a, name in zip([self.random_agent, agent], ["random", "DT"]):
            self._evaluate_agent_performance(
                a,
                name,
                dataset,
                dataset.train_config,
                "efficiency",
                self.train_seeds,
                state,
                control,
            )

    def _evaluate_iid_generalisation(self, dataset, agent, state: TrainerState):
        self._set_val_seeds()

        # run evaluations using both the agent being trained and a random agent (for baseline comparison)
        for a, name in zip([self.random_agent, agent], ["random", "DT"]):
            self._evaluate_agent_performance(
                a,
                name,
                dataset,
                dataset.train_config,
                "iid_generalisation",
                self.val_seeds,
                state,
            )

    def _evaluate_ood_generalisation(self, dataset, agent, state: TrainerState):
        per_config_repeats = self.num_repeats // len(dataset.test_configs) + (
            self.num_repeats % len(dataset.test_configs) > 0
        )

        n_seeds_sampled = 0
        for config in dataset.test_configs:
            # sample some seeds for the current config
            if n_seeds_sampled + per_config_repeats <= self.num_repeats:
                n_seeds_to_sample = per_config_repeats
            else:
                n_seeds_to_sample = self.num_repeats - n_seeds_sampled
            self._load_seed_dict(dataset, config)
            seeds = self._sample_test_seeds(n=n_seeds_to_sample)

            # run evaluations using both the agent being trained and a random agent (for baseline comparison)
            for a, name in zip([self.random_agent, agent], ["random", "DT"]):
                self._evaluate_agent_performance(
                    a, name, dataset, config, "ood_generalisation", seeds, state
                )

            n_seeds_sampled += n_seeds_to_sample

    def _set_val_seeds(self):
        val_seeds = self._sample_validation_seeds(self.num_repeats)
        self.val_seeds = val_seeds

    def _set_train_seeds(self):
        train_seeds = self.collator.dataset.train_seeds
        self.train_seeds = {"": train_seeds} if len(train_seeds) <= self.num_repeats else self._sample_train_seeds(train_seeds, self.num_repeats)

    def _load_seed_dict(self, dataset, config):
        self.seeds = dataset.seed_finder.load_seeds(dataset.level, config)

    def _sample_train_seeds(self, train_seeds, n=1):
        return {"": np.random.choice(train_seeds, size=n, replace=False)}

    def _sample_validation_seeds(self, n=1):
        if self.seeds is None:
            raise Exception("No seeds loaded")
        return {"": np.random.choice(self.seeds["validation_seeds"], size=n, replace=False)}

    def _sample_test_seeds(self, n=1):
        if self.seeds is None:
            raise Exception("No seeds loaded")

        seeds = {}
        types = [
            k
            for k in self.seeds
            if "seed" not in k and self.seeds[k]["test_seeds"] and k != "task_task"
        ]
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
        eval_type,
        ood_type,
        model_name,
        ret,
        ep_length,
        success,
        gc_success,
        global_step,
    ):
        self.results["model"].append(model_name)
        self.results["samples"].append(self.collator.samples_processed)
        self.results["level"].append(dataset.level)
        self.results["config"].append(config)
        self.results["seed"].append(seed)
        self.results["eval_type"].append(eval_type)
        self.results["ood_type"].append(ood_type)
        self.results["return"].append(ret)
        self.results["episode_length"].append(ep_length)
        self.results["success"].append(success)
        self.results["gc_success"].append(gc_success)
        self.results["pw_success"].append(
            self._get_pw_success(success, ep_length, dataset.level)
        )
        self.results["global_step"].append(global_step)

        df = pd.DataFrame(self.results)

        if self.user_args["record_video"]:
            env.release()

            if self.samples_processed == 0:
                env.save_as(f"first_{model_name}")

            if "generalisation" in eval_type and model_name == "DT":
                log(f"Current gc success {float(gc_success)}")
                if float(gc_success) == float(1):
                    log("saving video for successful episode")
                    env.save_as(
                        f"{config}_{seed}_{'ood_' + ood_type + '_' if 'ood' in eval_type else ''}succesful"
                    )
                if float(gc_success) == float(0):
                    log("saving video for failed episode")
                    env.save_as(
                        f"{config}_{seed}_{'ood_' + ood_type + '_' if 'ood' in eval_type else ''}failed"
                    )
                max_return = df[
                    (df["model"] == "DT")
                    & (df["eval_type"] == eval_type)
                    & (df["ood_type"] == ood_type)
                ]["return"].max()
                log(f"Current max return {max_return}")
                max_return = max_return or 0
                if ret > max_return:
                    log("saving video for new best episode")
                    env.save_as(
                        f"best_return_{eval_type}_{'ood_' + ood_type + '_' if 'ood' in eval_type else ''}"
                    )

    def _evaluate_agent_performance(
        self,
        agent: Agent,
        agent_name: str,
        dataset,
        config: str,
        eval_type: str,
        seeds: dict,
        state: TrainerState,
        control: TrainerControl = None,
    ):
        run_agent = self._run_agent_on_minigrid_env

        # for each repeat, run agent and record metrics (and optionally render a video of the episode)
        gc_successes = []
        for ood_type, seeds in seeds.items():  # ood_type is "" if not ood
            for seed in seeds:
                seed = int(seed)
                env = self._create_env(config, seed)
                ret, ep_length, success, gc_success = run_agent(
                    agent, env, seed, self.target_return
                )
                step = state.global_step if ood_type == "efficiency" else 99999
                gc_successes.append(gc_success)
                self._record_result(
                    env,
                    dataset,
                    config,
                    seed,
                    eval_type,
                    ood_type,
                    agent_name,
                    ret,
                    ep_length,
                    success,
                    gc_success,
                    step,
                )
        mean_gc_success = np.mean(gc_successes)
        early_stopping = False
        if eval_type == "efficiency" and agent_name == "DT":
            early_stopping = self._check_early_stopping(mean_gc_success, state, control)

        # convert results to dataframe
        df = pd.DataFrame(self.results)

        if eval_type == "efficiency" and early_stopping:
            df.drop(df[df["global_step"] > self.best_global_step].index, inplace=True)
            df.to_pickle(os.path.join(self.output_dir, "results.pkl"))
            control.should_training_stop = True

        # log the average episode return for the current eval
        # log(
        #     f"average return ({agent_name} agent) on level {dataset.level}: {df[df['samples'] == self.collator.samples_processed]['return'].mean()}",
        #     with_tqdm=True,
        # )

        # save the results to disk
        else:
            df.to_pickle(os.path.join(self.output_dir, "results.pkl"))

    def _check_early_stopping(
        self, mean_gc_success, state: TrainerState, control: TrainerControl
    ):
        self._check_metric_improvement(mean_gc_success, state)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            log(f"Early stopping at step {state.global_step}")
            return True
        return False

    def _check_metric_improvement(self, metric, state: TrainerState):
        if (
            metric > self.best_gc_success + self.early_stopping_threshold
            and state.global_step > 0
        ):
            log(
                f"Achieved new best gc success {metric} (higher than previous best gc success {self.best_gc_success} + threshold {self.early_stopping_threshold})"
            )
            self.best_gc_success = metric
            self.best_global_step = state.global_step
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1
            log(
                f"Increasing patience counter to {self.early_stopping_patience_counter} as gc success ({metric}) lower than current best gc success ({self.best_gc_success}) + threshold ({self.early_stopping_threshold})"
            )

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

        def get_mission_embeddings(obs):
            if self.user_args["mission_at_inference"] == "actual":
                return self.collator.embed_sentences(np.array([obs["mission"]] * self.user_args["context_length"]), "mission").to(self.device)
            if self.user_args["mission_at_inference"] == "str_constant":
                return self.str_mission_embeddings
            return self.int_mission_embeddings

        def get_feedback_embeddings():
            if self.user_args["feedback_at_inference"] == "actual":
                return NotImplementedError("Feedback at inference not implemented yet.")
            if self.user_args["feedback_at_inference"] == "str_constant":
                return self.str_feedback_embeddings
            return self.int_feedback_embeddings

        max_ep_len = env.max_steps
        obs, _ = env.reset(seed=seed)

        mission_embeddings = get_mission_embeddings(obs)
        feedback_embeddings = get_feedback_embeddings()

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
                    mission_embeddings=mission_embeddings[:, : t + 1, :],
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    returns_to_go=returns_to_go,
                    timesteps=timesteps,
                    feedback_embeddings=feedback_embeddings[:, : t + 1, :],
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
        eff_df = df[df["eval_type"] == "efficiency"]
        iid_gen_df = df[df["eval_type"] == "iid_generalisation"]
        ood_gen_df = df[df["eval_type"] == "ood_generalisation"]

        metrics = set(self.results.keys()).difference(
            {
                "samples",
                "model",
                "level",
                "config",
                "seed",
                "ood_type",
                "eval_type",
                "global_step",
            }
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
                fig.savefig(
                    os.path.join(self.output_dir, f"eff_{m.replace(' ', '_')}.{fmt}")
                )
            plt.close(fig)

            # then, do bar plot (for generalisation) (if we have data yet)
            # IID generalisation plot
            if len(iid_gen_df) > 0:
                fig, ax = plt.subplots()
                sns.barplot(x="model", y=m, data=iid_gen_df, ax=ax)
                for fmt in formats:
                    fig.savefig(
                        os.path.join(
                            self.output_dir, f"iid_gen_{m.replace(' ', '_')}.{fmt}"
                        )
                    )
                plt.close(fig)

            # OOD generalisation plots
            if len(ood_gen_df) > 0:
                fig, ax = plt.subplots()
                sns.barplot(x="model", y=m, hue="ood_type", data=ood_gen_df, ax=ax)
                for fmt in formats:
                    fig.savefig(
                        os.path.join(
                            self.output_dir, f"ood_gen_{m.replace(' ', '_')}.{fmt}"
                        )
                    )
                plt.close(fig)
