import os
import shutil

import gymnasium as gym
import numpy as np
from dopamine.replay_memory import circular_replay_buffer
from jsonc_parser.parser import JsoncParser
from tqdm import tqdm

from src.dataset.minari_dataset import MinariDataset
from src.dataset.minari_storage import name_dataset
from src.dataset.seeds import LEVELS_CONFIGS
from src.dataset.seeds import SeedFinder
from src.env.feedback_env import FeedbackEnv
from src.ppo.ppo_agent import PPOAgent
from src.utils.utils import discounted_cumsum
from src.utils.utils import get_minigrid_obs
from src.utils.utils import log
from src.utils.utils import normalise
from src.utils.utils import to_one_hot

from memory_profiler import profile


class CustomDataset:
    """
    Class for generating a custom dataset for a given environment, seed and policy.
    """

    def __init__(self, args):
        self.args = args
        self.shard = None
        self.buffers = []
        self.steps = []
        self.ep_counts = []
        self.seed_finder = SeedFinder(self.args["num_episodes"])
        self.level = self.args["level"]
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

    @profile
    def _get_dataset(self):
        """
        Get a MinariDataset object, either by loading an existing dataset from local storage
        or by generating a new dataset.

        Returns
        -------
        MinariDataset: the dataset object that was retrieved from storage or created.
        """
        dataset_name = name_dataset(self.args)
        minari_fp = os.environ.get("MINARI_DATASETS_PATH") or os.path.join(
            os.path.expanduser("~"), ".minari", "datasets"
        )
        self.fp, self.num_shards = os.path.join(minari_fp, dataset_name), 0

        if self.args["load_existing_dataset"] and os.path.exists(self.fp):
            print(f"Loading existing dataset {dataset_name}")
            self._load_dataset(self.fp)
        else:
            print(f"Creating dataset {dataset_name}")
            self._initialise_new_dataset()
            if self.args["policy"] == "random":
                self._generate_new_dataset()
            else:
                self._from_ppo_training()
            self.buffer = []
            self.steps = []
            self.ep_counts = []

        return self

    def _load_dataset(self, datset_dir):
        self.num_shards = 0
        for _ in os.listdir(datset_dir):
            self.num_shards += 1

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
        step_ceiling = 8**2 * 3**2

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
        # return min(global_max_steps, step_ceiling)
        return global_max_steps

    def _initialise_buffers(
        self, num_buffers, obs_shape, config=""
    ):
        log(f"initialising {num_buffers} buffers of size {self.args['eps_per_shard']}", with_tqdm=True)
        for _ in range(num_buffers):
            self.buffers.append(self._create_buffer(obs_shape, config))
            self.steps.append(0)
            self.ep_counts.append(0)

    def _create_buffer(self, obs_shape, config=""):
        num_eps = self.args["eps_per_shard"]
        max_steps = self._get_level_max_steps()
        return {
            "configs": [config] * ((max_steps + 1) * num_eps),
            "seeds": np.array([[0]] * ((max_steps + 1) * num_eps)),
            "missions": [self.env.get_mission()] * ((max_steps + 1) * num_eps),
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
            "feedback": [self.env.get_feedback_constant()]
            * ((max_steps + 1) * num_eps),
            "terminations": np.array([[0]] * ((max_steps + 1) * num_eps), dtype=bool),
            "truncations": np.array([[0]] * ((max_steps + 1) * num_eps), dtype=bool),
        }

    def _flush_buffer(self, buffer_idx, obs_shape, config=""):
        # if buffer exists and isn't empty, first save it to file
        if (
            len(self.buffers) > buffer_idx
            and len(self.buffers[buffer_idx]["observations"]) > 0
            and np.any(self.buffers[buffer_idx]["observations"])  # check any nonzero
        ):
            self._save_buffer_to_minari_file(buffer_idx)

        if obs_shape is None:
            obs_shape = self.buffers[buffer_idx]["observations"][0].shape

        self.buffers[buffer_idx] = self._create_buffer(obs_shape, config)
        self.steps[buffer_idx] = 0
        self.ep_counts[buffer_idx] = 0

    def _save_buffer_to_minari_file(self, buffer_idx):
        for key in self.buffers[buffer_idx].keys():
            self.buffers[buffer_idx][key] = self.buffers[buffer_idx][key][
                : self.steps[buffer_idx] + 2
            ]

        episode_terminals = (
            self.buffers[buffer_idx]["terminations"]
            + self.buffers[buffer_idx]["truncations"]
            if self.args["include_timeout"]
            else None
        )

        md = MinariDataset(
            level_group=self.category,
            level_name=self.args["level"],
            dataset_name=name_dataset(self.args),
            policy=self.args["policy"],
            feedback_mode=self.args["feedback_mode"],
            configs=self.buffers[buffer_idx]["configs"],
            seeds=self.buffers[buffer_idx]["seeds"],
            code_permalink="https://github.com/maxtaylordavies/feedback-DT/blob/master/src/_datasets.py",
            author="Sabrina McCallum",
            author_email="s2431177@ed.ac.uk",
            missions=self.buffers[buffer_idx]["missions"],
            observations=self.buffers[buffer_idx]["observations"],
            actions=self.buffers[buffer_idx]["actions"],
            rewards=self.buffers[buffer_idx]["rewards"],
            feedback=self.buffers[buffer_idx]["feedback"],
            terminations=self.buffers[buffer_idx]["terminations"],
            truncations=self.buffers[buffer_idx]["truncations"],
            episode_terminals=episode_terminals,
        )

        fp = os.path.join(self.fp, str(self.num_shards))
        log(
            f"writing buffer to file {fp}.hdf5 ({len(self.buffers[buffer_idx]['observations'])} steps)",
            with_tqdm=True,
        )

        md.save(fp)
        self.num_shards += 1

    def _add_to_buffer(
        self,
        buffer_idx,
        observation,
        action,
        reward,
        feedback,
        terminated,
        truncated,
        config,
        seed,
        mission,
    ):
        self.buffers[buffer_idx]["observations"][self.steps[buffer_idx]] = observation
        self.buffers[buffer_idx]["actions"][self.steps[buffer_idx]] = action
        self.buffers[buffer_idx]["rewards"][self.steps[buffer_idx]] = reward
        self.buffers[buffer_idx]["feedback"][self.steps[buffer_idx]] = feedback
        self.buffers[buffer_idx]["terminations"][self.steps[buffer_idx]] = terminated
        self.buffers[buffer_idx]["truncations"][self.steps] = truncated
        self.buffers[buffer_idx]["configs"][self.steps[buffer_idx]] = config
        self.buffers[buffer_idx]["seeds"][self.steps[buffer_idx]] = seed
        self.buffers[buffer_idx]["missions"][self.steps[buffer_idx]] = mission

        self.steps[buffer_idx] += 1
        if terminated or truncated:
            self.ep_counts[buffer_idx] += 1

    def _create_episode(self, config, seed, buffer_idx=0):
        partial_obs, _ = self.env.reset(seed=seed)
        terminated, truncated = False, False
        while not (terminated or truncated):
            obs = get_minigrid_obs(
                self.env, partial_obs, self.args["fully_obs"], self.args["rgb_obs"]
            )
            mission = partial_obs["mission"]
            action = np.random.randint(0, 6)  # random policy

            # execute action
            partial_obs, reward, terminated, truncated, feedback = self.env.step(action)
            reward = (
                feedback if self.args["feedback_mode"] == "numerical_reward" else reward
            )

            self._add_to_buffer(
                buffer_idx=buffer_idx,
                observation=obs["image"],
                action=action,
                reward=reward,
                feedback=feedback,
                terminated=terminated,
                truncated=truncated,
                config=config,
                seed=seed,
                mission=mission,
            )

    def _initialise_new_dataset(self):
        # create folder to store MinariDataset files
        if os.path.exists(self.fp):
            print("Overwriting existing dataset folder")
            shutil.rmtree(self.fp, ignore_errors=True)
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
                    self.env = FeedbackEnv(gym.make(config), self.args["feedback_mode"])
                    partial_obs, _ = self.env.reset(seed=seed)
                    obs = get_minigrid_obs(
                        self.env,
                        partial_obs,
                        self.args["fully_obs"],
                        self.args["rgb_obs"],
                    )["image"]

                    self.state_dim = np.prod(obs.shape)

                    # initialise buffers to store replay data
                    if current_episode == 0:
                        self._initialise_buffers(
                            num_buffers=1,
                            obs_shape=obs.shape,
                            config=config,
                        )

                    # create another episode
                    self._create_episode(config, seed)

                    # if buffer contains 1000 episodes or this is final episode, save data to file and clear buffer
                    if (
                        current_episode > 0 and current_episode % self.args["eps_per_shard"] == 0
                    ) or (current_episode == self.args["num_episodes"] - 1):
                        self._flush_buffer(
                            buffer_idx=0, obs_shape=obs.shape, config=config
                        )

                    current_episode += 1
                    current_conf_episode += 1
                    pbar.update(1)
                    pbar.refresh()

        if hasattr(self, "env"):
            self.env.close()
        self._flush_buffer(buffer_idx=0, obs_shape=obs.shape)

    def load_shard(self, idx=None):
        if not idx:
            idx = np.random.randint(0, self.num_shards)
        self.shard = MinariDataset.load(os.path.join(self.fp, str(idx)))

        # compute start and end timesteps for each episode
        self.episode_ends = np.where(
            self.shard.terminations + self.shard.truncations == 1
        )[0]
        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1] + 1])
        self.episode_lengths = self.episode_ends - self.episode_starts + 1
        self.num_episodes = len(self.episode_starts)

        # store state and action dimensions
        self.state_dim, self.act_dim = (
            np.prod(self.shard.observations.shape[1:]),
            6,
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
            start += (
                np.random.randint(0, self.episode_lengths[ep_idx] - 1)
                if self.episode_lengths[ep_idx] > 1
                else 0
            )
        tmp = start + length - 1 if length else self.episode_ends[ep_idx]
        end = min(tmp, self.episode_ends[ep_idx]) + 1
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

    def _from_ppo_training(self):
        # compute number of eps to sample per config
        episodes_per_config = (self.args["num_episodes"] // len(self.configs)) + (
            self.args["num_episodes"] % len(self.configs) > 0
        )

        # helper func to set up buffer for env
        def setup(env, config, num_seeds):
            self.env = env
            partial_obs, _ = self.env.reset(seed=0)
            obs = get_minigrid_obs(
                self.env,
                partial_obs,
                self.args["fully_obs"],
                self.args["rgb_obs"],
            )["image"]
            self._initialise_buffers(
                num_buffers=num_seeds, obs_shape=obs.shape, config=config
            )
            self.config_eps = 0

        # define callback func for storing data
        def callback(exps, logs, config, seeds):
            obss = exps.obs.image.cpu().numpy()
            actions = exps.action.cpu().numpy().reshape(-1, 1)
            rewards = exps.reward.cpu().numpy().reshape(-1, 1)
            feedback = exps.feedback.reshape(-1, 1)  # feedback is already a numpy array

            # they don't provide terminations/truncations - but mask is computed as 1 - (terminated or truncated)
            # so we'll just assume all zero values of mask correspond to terminations (and ignore truncations)
            terminations = 1 - exps.mask.cpu().numpy()

            # reshape tensors to be (num_seeds, num_timesteps_per_seed, ...)
            tensors = [obss, actions, rewards, feedback, terminations]
            for i in range(len(tensors)):
                tensors[i] = tensors[i].reshape(len(seeds), -1, *tensors[i].shape[1:])
            obss, actions, rewards, feedback, terminations = tensors

            for i, seed in enumerate(seeds):
                for t in range(obss.shape[1]):
                    # process partial observation
                    o = get_minigrid_obs(
                        self.env,
                        obss[i, t],
                        self.args["fully_obs"],
                        self.args["rgb_obs"],
                    )["image"]

                    # determine reward
                    r = (
                        feedback[i, t]
                        if self.args["feedback_mode"] == "numerical_reward"
                        else rewards[i, t]
                    )

                    # add step to buffer i
                    self._add_to_buffer(
                        buffer_idx=i,
                        action=actions[i, t],
                        observation=o,
                        reward=r,
                        feedback=feedback[i, t],
                        terminated=terminations[i, t],
                        truncated=0,
                        config=config,
                        seed=seed,
                        mission=self.env.get_mission(),
                    )

                    # if buffer i is full, flush it
                    if terminations[i, t] and self.ep_counts[i] >= self.args["eps_per_shard"]:
                        self._flush_buffer(
                            buffer_idx=i, obs_shape=o.shape, config=config
                        )

            # number of new episodes = number of nonzero elements in terminations
            self.config_eps += np.count_nonzero(terminations)

            # return True if we've collected enough episodes for this config
            return self.config_eps >= episodes_per_config

        # train a PPO agent for each config
        for config in tqdm(self.configs):
            log(f"config: {config}", with_tqdm=True)

            # choose random subset of train seeds for this config
            seed_log = self.seed_finder.load_seeds(self.args["level"], config)
            train_seeds = self.seed_finder.get_train_seeds(seed_log)
            seeds = [
                int(s)  # from np.int64
                for s in np.random.choice(train_seeds, size=128, replace=True)
            ]

            log(f"using seeds: {seeds}", with_tqdm=True)

            # train PPO agent
            ppo = PPOAgent(env_name=config, seeds=seeds)
            setup(ppo.env, config, len(seeds))
            ppo._train_agent(callback=callback)

            # flush any remaining data to file
            for i in range(len(self.buffers)):
                if self.ep_counts[i] > 0:
                    self._save_buffer_to_minari_file(i)

        # return dataset
        return self

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
