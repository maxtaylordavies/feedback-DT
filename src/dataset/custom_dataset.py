import os
import shutil

import gymnasium as gym
import numpy as np
from dopamine.replay_memory import circular_replay_buffer
from jsonc_parser.parser import JsoncParser
from tqdm import tqdm

from src.dataset.custom_feedback_verifier import (
    RandomFeedback,
    RuleFeedback,
    TaskFeedback,
)
from src.dataset.minari_dataset import MinariDataset
from src.dataset.minari_storage import list_local_datasets, name_dataset
from src.dataset.seeds import LEVELS_CONFIGS, SeedFinder
from src.utils.ppo import PPOAgent
from src.utils.utils import (
    discounted_cumsum,
    get_minigrid_obs,
    log,
    normalise,
    to_one_hot,
)

EPS_PER_SHARD = 100


class CustomDataset:
    """
    Class for generating a custom dataset for a given environment, seed and policy.
    """

    def __init__(self, args):
        self.args = args
        self.shard = None
        self.buffer = None
        self.seed_finder = SeedFinder(self.args["num_episodes"])
        self.configs = self._get_configs()
        self.category = self._get_category()

    def _get_configs(self):
        """
        Get the configs for the given level that are suitable for training.

        Check, for each possible config for the level, if the number of safe train seeds is non-zero and include only those.

        Returns
        -------
            list: the configs.
        """
        configs = []
        for config in LEVELS_CONFIGS["original_tasks"][self.args["level"]]:
            seed_log = self.seed_finder.load_seeds(self.args["level"], config)
            if seed_log["n_train_seeds"]:
                configs.append(config)
        return configs

    def _get_dataset(self):
        """
        Get a MinariDataset object, either by loading an existing dataset from local storage
        or by generating a new dataset.

        Returns
        -------
        MinariDataset: the dataset object that was retrieved from storage or created.
        """
        dataset_name = name_dataset(self.args)
        print(f"Creating dataset {dataset_name}")
        self._initialise_new_dataset()
        self._generate_new_dataset()
        return self

        # if self.args["load_dataset_if_exists"] and dataset_name in list_local_datasets():
        #     log(f"Loading dataset {dataset_name} from local storage")
        #     return MinariDataset.load(dataset_name)
        # else:
        #     dataset = self._generate_new_dataset()
        #     log(f"Created new dataset {dataset_name}, saving to local storage")
        #     dataset.save()
        #     return dataset

    def _get_category(self):
        """
        Get the category from the level.

        Returns
        -------
        str: the category.
        """
        metadata_path = os.getenv("ENV_METADATA_PATH", "env_metadata.jsonc")
        metadata = JsoncParser.parse_file(metadata_path)["levels"]
        for level_group, levels in metadata.items():
            if self.args["level"] in levels:
                return level_group

    def _get_used_action_space(self):
        """
        Get the used action space for the environment.

        Returns
        -------
        list: the used action space.
        """
        metadata_path = os.getenv("ENV_METADATA_PATH", "env_metadata.jsonc")
        metadata = JsoncParser.parse_file(metadata_path)["levels"]
        return metadata[self.category][self.args["level"]]["used_action_space"]

    def _get_level_max_steps(self):
        """
        Get the max steps for the environment.

        Returns
        -------
        int: the max steps.
        """
        metadata_path = os.getenv("ENV_METADATA_PATH", "env_metadata.jsonc")
        metadata = JsoncParser.parse_file(metadata_path)
        level_metadata = metadata["levels"][self.category][self.args["level"]]

        seq_instrs_factor = 4 if level_metadata["mission_space"]["sequence"] else 1
        putnext_instrs_factor = 2 if level_metadata["putnext"] else 1
        max_instrs_factor = 1 * seq_instrs_factor * putnext_instrs_factor

        global_max_steps = 0
        for config in self.configs:
            try:
                max_steps = level_metadata[config]["max_steps"]
            except KeyError:
                try:
                    room_size = level_metadata[config]["room_size"]
                except KeyError:
                    room_size = metadata["defaults"]["room"]["room_size"]
                try:
                    num_rows = level_metadata[config]["num_rows"]
                except KeyError:
                    num_rows = (
                        metadata["defaults"]["maze"]["num_rows"]
                        if level_metadata["maze"]
                        else 1
                    )
                try:
                    num_cols = level_metadata[config]["num_cols"]
                except KeyError:
                    num_cols = (
                        metadata["defaults"]["maze"]["num_cols"]
                        if level_metadata["maze"]
                        else 1
                    )
                max_steps = room_size**2 * num_rows * num_cols * max_instrs_factor
            global_max_steps = max(global_max_steps, max_steps)
        return global_max_steps

    def _policy(self, observation):
        """
        Get the next action from a given policy.

        Parameters
        ----------
        observation (np.ndarray): the observation.

        Returns
        -------
        int: the next action.
        """

        if self.args["policy"] == "random_used_action_space_only":
            return np.random.choice(self._get_used_action_space())
        if "ppo" in self.args["policy"]:
            raise NotImplementedError
        # Excluding the 'done' action (integer representation: 6), as by default, this is not used
        # to evaluate success for any of the tasks
        return np.random.randint(0, 6)

    def _create_feedback_verifiers(self):
        self.rule_feedback_verifier = RuleFeedback()
        self.task_feedback_verifier = TaskFeedback(self.env)
        self.random_feedback_verifier = RandomFeedback(
            "lorem_ipsum"
            if "lorem_ipsum" in self.args["feedback_mode"]
            else "random_sentence"
        )

    def _get_feedback_constant(self):
        """
        Get the constant feedback string depending on the feedback mode.

        Returns
        -------
            str: the constant feedback string.
        """
        if self.args["feedback_mode"] == "random_lorem_ipsum":
            return "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        if self.args["feedback_mode"] == "numerical_reward":
            return np.array(0, dtype=np.float32)
        return "No feedback available."

    def _get_feedback(self, action):
        """
        Get the feedback for a given action.

        Parameters
        ----------
            action (int): the action.

        Returns
        -------
            str: the feedback.
        """
        rule_feedback = (
            self.rule_feedback_verifier.verify_feedback(self.env, action)
            if self.args["feedback_mode"] in ["all", "rule_only"]
            else None
        )
        task_feedback = (
            self.task_feedback_verifier.verify_feedback(self.env, action)
            if self.args["feedback_mode"] in ["all", "task_only"]
            else None
        )

        if self.args["feedback_mode"] == "random":
            return self.random_feedback_verifier.verify_feedback()
        if self.args["feedback_mode"] == "rule_only":
            return rule_feedback
        if self.args["feedback_mode"] == "task_only":
            return task_feedback
        if self.args["feedback_mode"] == "numerical_reward":
            if task_feedback != "No feedback available.":
                return np.array(1)
            if rule_feedback != "No feedback available.":
                return np.array(-1)
            return np.array(0)
        if self.args["feedback_mode"] == "all":
            if rule_feedback == "No feedback available.":
                return task_feedback
            return rule_feedback

    def _save_buffer_to_minari_file(self):
        for key in self.buffer.keys():
            self.buffer[key] = self.buffer[key][: self.steps + 2]

        episode_terminals = (
            self.buffer["terminations"] + self.buffer["truncations"]
            if self.args["include_timeout"]
            else None
        )

        md = MinariDataset(
            level_group=self.category,
            level_name=self.args["level"],
            dataset_name=name_dataset(self.args),
            policy=self.args["policy"],
            feedback_mode=self.args["feedback_mode"],
            configs=self.buffer["configs"],
            seeds=self.buffer["seeds"],
            code_permalink="https://github.com/maxtaylordavies/feedback-DT/blob/master/src/_datasets.py",
            author="Sabrina McCallum",
            author_email="s2431177@ed.ac.uk",
            missions=self.buffer["missions"],
            observations=self.buffer["observations"],
            actions=self.buffer["actions"],
            rewards=self.buffer["rewards"],
            feedback=self.buffer["feedback"],
            terminations=self.buffer["terminations"],
            truncations=self.buffer["truncations"],
            episode_terminals=episode_terminals,
        )

        fp = os.path.join(self.fp, str(self.num_shards))
        log(
            f"writing buffer to file {fp}.hdf5 ({len(self.buffer['observations'])} steps)",
            with_tqdm=True,
        )
        md.save(fp)
        self.num_shards += 1

    def _flush_buffer(self, obs_shape, config="", num_eps=EPS_PER_SHARD):
        # if buffer exists and isn't empty, first save it to file
        if (
            self.buffer
            and len(self.buffer["observations"]) > 0
            and np.any(self.buffer["observations"])  # check any nonzero
        ):
            print("saving buffer to file")
            self._save_buffer_to_minari_file()

        max_steps = self._get_level_max_steps()
        self.buffer = {
            "configs": [config] * ((max_steps + 1) * num_eps),
            "seeds": np.array([[0]] * ((max_steps + 1) * num_eps)),
            "missions": [self.env.instrs.surface(self.env)] * ((max_steps + 1) * num_eps),
            "observations": np.array(
                [np.zeros(obs_shape)] * ((max_steps + 1) * num_eps),
                dtype=np.uint8,
            ),
            "actions": np.array(
                [[0]] * ((max_steps + 1) * num_eps),
                dtype=np.float32,
            ),
            "rewards": np.array(
                [[0]] * ((max_steps + 1) * num_eps),
                dtype=np.float32,
            ),
            "feedback": [self._get_feedback_constant()] * ((max_steps + 1) * num_eps),
            "terminations": np.array([[0]] * ((max_steps + 1) * num_eps), dtype=bool),
            "truncations": np.array([[0]] * ((max_steps + 1) * num_eps), dtype=bool),
        }
        self.steps = 0

    def _add_to_buffer(
        self,
        action,
        observation,
        reward,
        feedback,
        terminated,
        truncated,
        config,
        seed,
    ):
        # store action a_t taken after observing o_t
        self.buffer["actions"][self.steps] = action

        # store obs o_t+1, reward r_t+1 etc resulting from taking a_t at o_t
        self.steps += 1
        self.buffer["observations"][self.steps] = observation
        self.buffer["rewards"][self.steps] = reward
        self.buffer["feedback"][self.steps] = feedback
        self.buffer["terminations"][self.steps] = terminated
        self.buffer["truncations"][self.steps] = truncated
        self.buffer["configs"][self.steps] = config
        self.buffer["seeds"][self.steps] = seed

    def _create_episode(self, config, seed):
        partial_obs, _ = self.env.reset(seed=seed)
        # Storing observation at initial episode timestep t=0 (o_0)
        obs = get_minigrid_obs(
            self.env, partial_obs, self.args["fully_obs"], self.args["rgb_obs"]
        )
        self.buffer["observations"][self.steps] = obs["image"]
        terminated, truncated = False, False
        while not (terminated or truncated):
            # Passing partial observation to policy (PPO) as agent was trained on this
            # following the original implementation
            action = self._policy(partial_obs)
            partial_obs, reward, terminated, truncated, _ = self.env.step(action)
            obs = get_minigrid_obs(
                self.env, partial_obs, self.args["fully_obs"], self.args["rgb_obs"]
            )
            feedback = self._get_feedback(action)
            reward = feedback if self.args["feedback_mode"] == "numerical_reward" else reward

            self._add_to_buffer(
                action=action,
                observation=obs["image"],
                reward=reward,
                feedback=feedback,
                terminated=terminated,
                truncated=truncated,
                config=config,
                seed=seed,
            )

    def _initialise_new_dataset(self):
        # create folder to store MinariDataset files
        fp = os.environ.get("MINARI_DATASETS_PATH") or os.path.join(
            os.path.expanduser("~"), ".minari", "datasets"
        )
        self.fp, self.num_shards = os.path.join(fp, name_dataset(self.args)), 0
        if os.path.exists(self.fp):
            shutil.rmtree(self.fp)
        os.makedirs(self.fp)

    def _generate_new_dataset(self):
        pbar = tqdm(total=self.args["num_episodes"], desc="Generating dataset")

        episodes_per_config = (self.args["num_episodes"] // len(self.configs)) + (
            self.args["num_episodes"] % len(self.configs) > 0
        )
        current_episode = 0

        for config in self.configs:
            log(f"config: {config}", with_tqdm=True)

            seed_log = self.seed_finder.load_seeds(self.args["level"], config)
            train_seeds = self.seed_finder.get_train_seeds(seed_log)
            log(f"num train seeds: {len(train_seeds)}", with_tqdm=True)

            current_conf_episode, done = 0, False
            while not done:
                for seed in train_seeds:
                    seed = int(seed)  # from np.int64

                    done = (
                        current_conf_episode >= episodes_per_config
                        or current_episode >= self.args["num_episodes"]
                    )

                    if done or seed >= seed_log["last_seed_tested"]:
                        # the second condition is here in case we want to use the same set of training seeds multiple times
                        # e.g. if we want to create really big random dataset
                        # and likely this will be necessary to train a PPO agent to convergence for harder levels
                        # this should not result in duplicate trajectories despite the same seeds being used multiple times
                        # when used with random policy, and even when used with the PPO agent as the agent will be at different
                        # model checkpoints at the time the seeds get repeated
                        break

                    # create and initialise environment
                    log("creating env", with_tqdm=True)
                    self.env = gym.make(config)
                    partial_obs, _ = self.env.reset(seed=seed)
                    obs = get_minigrid_obs(
                        self.env,
                        partial_obs,
                        self.args["fully_obs"],
                        self.args["rgb_obs"],
                    )["image"]

                    self.state_dim = np.prod(obs.shape)
                    log(f"state_dim: {self.state_dim}")

                    # initialise buffer to store replay data
                    if current_episode == 0:
                        log("initialising buffer", with_tqdm=True)
                        self._flush_buffer(obs.shape, config)

                    # feedback verifiers
                    self._create_feedback_verifiers()

                    # create another episode
                    self._create_episode(config, seed)

                    # if buffer contains 1000 episodes or this is final episode, save data to file and clear buffer
                    if (current_episode > 0 and current_episode % EPS_PER_SHARD == 0) or (
                        current_episode == self.args["num_episodes"] - 1
                    ):
                        self._flush_buffer(obs.shape, config)

                    current_episode += 1
                    current_conf_episode += 1
                    pbar.update(1)
                    pbar.refresh()

        self.env.close()
        self._flush_buffer(obs.shape)

    def load_shard(self, idx=None):
        if not idx:
            idx = np.random.randint(0, self.num_shards)
        self.shard = MinariDataset.load(os.path.join(self.fp, str(idx)))

        # compute start and end timesteps for each episode
        self.episode_ends = np.where(self.shard.terminations + self.shard.truncations == 1)[0]
        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1] + 1])
        self.episode_lengths = self.episode_ends - self.episode_starts + 1
        self.num_episodes = len(self.episode_starts)

        # store state and action dimensions
        self.state_dim, self.act_dim = (
            np.prod(self.shard.observations.shape[1:]),
            self.shard.get_action_size(),
        )

    def sample_episode_indices(self, num_eps, dist="uniform"):
        if not self.shard:
            raise Exception("No shard loaded")

        # define a distribution over episodes in current shard
        if dist == "length":
            probs = self.episode_lengths
        elif dist == "inverse_length":
            probs = 1 / self.episode_lengths
        else:
            probs = np.ones(self.num_episodes)
        probs /= np.sum(probs)

        # then use this distribution to sample episode indices
        return np.random.choice(
            np.arange(self.num_episodes),
            size=num_eps,
            p=probs,
        )

    def sample_episode(
        self, ep_idx, gamma, length=None, random_start=True, feedback=True, mission=True
    ):
        if not self.shard:
            raise Exception("No shard loaded")

        # optionally sample a random start timestep for this episode
        start = self.episode_starts[ep_idx]
        if random_start:
            start += np.random.randint(
                0,
                self.episode_lengths[ep_idx] - max(self.episode_lengths[ep_idx] // 4, 1),
            )
        tmp = start + length if length else self.episode_ends[ep_idx]
        end = min(tmp, self.episode_ends[ep_idx])
        s = self.shard.observations[start:end]
        s = normalise(s).reshape(1, -1, self.state_dim)

        a = self.shard.actions[start:end]
        a = to_one_hot(a, self.act_dim).reshape(1, -1, self.act_dim)

        rtg = discounted_cumsum(
            self.shard.rewards[start : self.episode_ends[ep_idx]], gamma=gamma
        )
        rtg = rtg[: end - start].reshape(1, -1, 1)

        f = (
            np.hstack(self.shard.feedback[start:end])
            if feedback
            else np.array(["No feedback available."] * s.shape[1])
        ).reshape(1, -1, 1)

        m = (
            np.hstack(self.shard.missions[start:end])
            if mission
            else np.array(["No mission available."] * s.shape[1])
        ).reshape(1, -1, 1)

        return {
            "timesteps": np.arange(0, end - start).reshape(1, -1),
            "mission": m,
            "states": s,
            "actions": a,
            "rewards": self.shard.rewards[start:end].reshape(1, -1, 1),
            "returns_to_go": rtg,
            "feedback": f,
            "attention_mask": np.ones((1, end - start)),
        }

    def __len__(self):
        return self.args["num_episodes"]

    # ----- these methods aren't used, but need to be defined for torch dataloaders to work -----

    def __getitem__(self, idx):
        return idx

    def __getitems__(self, idxs):
        return idxs

    # -------------------------------------------------------------------------------------------

    @classmethod
    def get_dataset(cls, args):
        print("Generating dataset...")
        return cls(args)._get_dataset()

    @classmethod
    def random(cls, num_eps, ep_length, state_dim, act_dim):
        states = np.random.rand(num_eps * ep_length, state_dim)
        actions = np.random.randint(0, act_dim, size=(num_eps * ep_length))
        rewards = np.random.rand(num_eps * ep_length)

        terminations = np.zeros((num_eps, ep_length))
        terminations[:, -1] = 1
        terminations = terminations.reshape((num_eps * ep_length))
        truncations = np.zeros_like(terminations)

        return cls(
            level_group="",
            level_name="",
            dataset_name="",
            policy="",
            feedback_mode="",
            configs=np.array([]),
            seeds=np.array([]),
            code_permalink="",
            author="",
            author_email="",
            missions=np.array([]),
            observations=states,
            actions=actions,
            rewards=rewards,
            feedback=np.array([]),
            terminations=terminations,
            truncations=truncations,
            episode_terminals=None,
            discrete_action=True,
        )

    @classmethod
    def from_dqn_replay(cls, data_dir, game, num_samples):
        obs, acts, rewards, dones = [], [], [], []

        buffer_idx, depleted = -1, True
        while len(obs) < num_samples:
            if depleted:
                buffer_idx, depleted = buffer_idx + 1, False
                buffer, i = load_dopamine_buffer(data_dir, game, 50 - buffer_idx), 0

            (
                s,
                a,
                r,
                _,
                _,
                _,
                terminal,
                _,
            ) = buffer.sample_transition_batch(batch_size=1, indices=[i])

            obs.append(s[0])
            acts.append(a[0])
            rewards.append(r[0])
            dones.append(terminal[0])

            i += 1
            depleted = i == buffer._replay_capacity

        return cls(
            level_group="",
            level_name="",
            dataset_name=f"dqn_replay-{game}-{num_samples}",
            policy="",
            feedback_mode="",
            configs=np.array([]),
            seeds=np.array([]),
            code_permalink="",
            author="",
            author_email="",
            missions=np.array([]),
            observations=np.array(obs),
            actions=np.array(acts),
            rewards=np.array(rewards),
            feedback=np.array([]),
            terminations=np.array(dones),
            truncations=np.zeros_like(dones),
            episode_terminals=None,
            discrete_action=True,
        )

    @classmethod
    def from_ppo_training(cls, args):
        # initialise dataset
        d = cls(args)

        # helper func to set up buffer for env
        def setup(env):
            d.env = env
            d._create_feedback_verifiers()
            partial_obs, _ = d.env.reset(seed=args["seed"])
            obs = get_minigrid_obs(
                d.env,
                partial_obs,
                d.args["fully_obs"],
                d.args["rgb_obs"],
            )["image"]
            d._flush_buffer(config=config, obs_shape=obs.shape)

        # define callback func for storing data
        def callback(exps, logs, config, seed):
            obss = exps.obs.image.cpu().numpy()
            actions = exps.action.cpu().numpy().reshape(-1, 1)
            rewards = exps.reward.cpu().numpy().reshape(-1, 1)

            # they don't provide terminations/truncations - but mask is computed as 1 - (terminated or truncated)
            # so we'll just assume all zero values of mask correspond to terminations (and ignore truncations)
            terminations = 1 - exps.mask.cpu().numpy()

            num_timesteps = obss.shape[0]
            for t in range(num_timesteps):
                o = get_minigrid_obs(  # process partial observation
                    d.env, obss[t], args["fully_obs"], args["rgb_obs"]
                )["image"]
                f = d._get_feedback(actions[t])
                r = f if args["feedback_mode"] == "numerical_reward" else rewards[t]
                d._add_to_buffer(
                    action=actions[t],
                    observation=o,
                    reward=r,
                    feedback=f,
                    terminated=terminations[t],
                    truncated=0,
                    config=config,
                    seed=seed,
                )

        # train a PPO agent for each config
        for config in tqdm(d.configs):
            log(f"config: {config}", with_tqdm=True)

            # choose random seed from train seeds
            seed_log = d.seed_finder.load_seeds(d.args["level"], config)
            train_seeds = d.seed_finder.get_train_seeds(seed_log)
            seed = int(np.random.choice(train_seeds))
            log(f"using seed {seed}", with_tqdm=True)

            # train PPO agent
            ppo = PPOAgent(env_name=config, seed=seed, n_frames=args["ppo_frames"])
            setup(ppo.env)
            ppo._train_agent(callback=callback)

        # return dataset
        return d


# helper func to load a dopamine buffer from dqn replay logs
def load_dopamine_buffer(data_dir, game, buffer_idx):
    replay_buffer = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=(84, 84),
        stack_size=4,
        update_horizon=1,
        gamma=0.99,
        observation_dtype=np.uint8,
        batch_size=32,
        replay_capacity=100000,
    )
    replay_buffer.load(os.path.join(data_dir, game, "1", "replay_logs"), buffer_idx)
    return replay_buffer
