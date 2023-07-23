import re
import os
import shutil

import gymnasium as gym
import numpy as np
from dopamine.replay_memory import circular_replay_buffer
from jsonc_parser.parser import JsoncParser
from tqdm import tqdm

from src.dataset.custom_feedback_verifier import RuleFeedback, TaskFeedback
from src.dataset.minari_dataset import MinariDataset
from src.dataset.minari_storage import list_local_datasets, name_dataset
from src.utils.utils import (
    log,
    get_minigrid_obs,
    discounted_cumsum,
    to_one_hot,
    normalise,
)

# from src.utils.ppo import PPOAgent

EPS_PER_SHARD = 100


class CustomDataset:
    """
    Class for generating a custom dataset for a given environment, seed and policy.
    """

    def __init__(self, args):
        self.args = args
        self.shard = None
        # if "ppo" in self.args["policy"]:
        #     ppo_agent = PPOAgent(
        #         self.args["env_name"], self.args["seed"], self.args["ppo_frames"]
        #     )
        #     self.ppo_model = ppo_agent.model

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

    def _get_level(self):
        """
        Get the level name from the environment name.

        Returns
        -------
        str: the level name.
        """
        temp_level = re.sub(r"([SRN]\d).*", r"", self.args["env_name"].split("-")[1])
        if temp_level.startswith("GoTo"):
            level = re.sub(r"(Open|ObjMaze|ObjMazeOpen)", r"", temp_level)
        elif temp_level.startswith("Open"):
            if "RedBlue" in temp_level:
                level = re.sub(r"(RedBlue)", r"Two", temp_level)
            else:
                level = re.sub(r"(Color|Loc)", r"", temp_level)
        elif temp_level.startswith("Unlock"):
            level = re.sub(r"(Dist)", r"", temp_level)
        elif temp_level.startswith("Pickup"):
            level = re.sub(r"(Dist)", r"", temp_level)
        else:
            level = temp_level
        return level

    def _get_category(self, level):
        """
        Get the category from the level name.

        Returns
        -------
        str: the category.
        """
        if level.startswith("GoTo"):
            return "GoTo"
        elif level.startswith("Open"):
            return "Open"
        elif level.startswith("Pickup") or level.startswith("UnblockPickup"):
            return "Pickup"
        elif level.startswith("PutNext"):
            return "PutNext"
        elif (
            level.startswith("Unlock")
            or level.startswith("KeyInBox")
            or level.startswith("BlockedUnlockPickup")
        ):
            return "Unlock"
        elif level.startswith("Synth") or "Boss" in level:
            return "Synth"
        else:
            return "Other"

    def _get_used_action_space(self):
        """
        Get the used action space for the environment.

        Returns
        -------
        list: the used action space.
        """
        file_path = os.getenv("ENV_METADATA_PATH", "env_metadata.jsonc")
        metadata = JsoncParser.parse_file(file_path)
        level = self._get_level()
        category = self._get_category(level)
        return metadata["levels"][category][level]["used_action_space"]

    # def _ppo(self, observation):
    #     """
    #     Get the next action from the PPO policy.

    #     Parameters
    #     ----------
    #     observation (np.ndarray): the (partial symbolic) observation.

    #     Reutrns
    #     ------
    #     int: the next action.
    #     """
    #     return self.ppo_model.get_action(observation)

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
        elif "ppo" in self.args["policy"]:
            raise NotImplementedError
            # return self._ppo(observation)
        else:
            # Excluding the 'done' action (integer representation: 6), as by default, this is not used
            # to evaluate success for any of the tasks
            return np.random.randint(0, 6)

    def _clear_buffer(self, obs_shape, num_eps=EPS_PER_SHARD):
        max_steps = self.env.max_steps
        self.buffer = {
            "missions": [""] * (max_steps * num_eps + 1),
            "observations": np.array(
                [np.zeros(obs_shape)] * (max_steps * num_eps + 1),
                dtype=np.uint8,
            ),
            "actions": np.array(
                [[0]] * (max_steps * num_eps + 1),
                dtype=np.float32,
            ),
            "rewards": np.array(
                [[0]] * (max_steps * num_eps + 1),
                dtype=np.float32,
            ),
            "feedback": ["TEST"] * (max_steps * num_eps + 1),
            "terminations": np.array([[0]] * (max_steps * num_eps + 1), dtype=bool),
            "truncations": np.array([[0]] * (max_steps * num_eps + 1), dtype=bool),
        }
        self.steps = 0

    def _create_episode(self):
        partial_obs, _ = self.env.reset(seed=self.args["seed"])

        # Storing mission for initial episode timestep t=0 (m_0)
        # (mission is the same for the whole episode)
        mission = partial_obs["mission"]
        self.buffer["missions"][self.steps] = mission

        # Storing observation at initial episode timestep t=0 (o_0)
        obs = get_minigrid_obs(
            self.env, partial_obs, self.args["fully_obs"], self.args["rgb_obs"]
        )
        self.buffer["observations"][self.steps] = obs["image"]

        terminated, truncated = False, False

        # Storing initial values for rewards, terminations, truncations and feedback
        self.buffer["rewards"][self.steps] = np.array(0)
        self.buffer["feedback"][self.steps] = "No feedback available."
        self.buffer["terminations"][self.steps] = np.array(terminated)
        self.buffer["truncations"][self.steps] = np.array(truncated)

        while not (terminated or truncated):
            action = self._policy(obs)
            feedback = self.rule_feedback_verifier.verify_feedback(self.env, action)
            partial_obs, reward, terminated, truncated, _ = self.env.step(action)

            # Storing action a_t taken after observing o_t
            self.buffer["actions"][self.steps] = np.array(action)

            # Generating and storing feedback f_t+1 resulting from taking a_t at o_t
            # If no rule feedback is provided (no rules have been breached), then
            # we set the feedback to be the task feedback, otherwise we set it to be
            # the rule feedback
            # (note that there should always either be rule feedback or task success feedback
            # as task success and rule violations are mutually exclusive)
            if feedback == "No feedback available.":
                feedback = self.task_feedback_verifier.verify_feedback(self.env, action)

            # Storing observation o_t+1, reward r_t+1, termination r_t+1, truncation r_t+1
            # resulting from taking a_t at o_t
            obs = get_minigrid_obs(
                self.env, partial_obs, self.args["fully_obs"], self.args["rgb_obs"]
            )
            self.buffer["observations"][self.steps + 1] = obs["image"]
            self.buffer["missions"][self.steps + 1] = mission
            self.buffer["rewards"][self.steps + 1] = np.array(reward)
            self.buffer["terminations"][self.steps + 1] = np.array(terminated)
            self.buffer["truncations"][self.steps + 1] = np.array(truncated)
            self.buffer["feedback"][self.steps + 1] = feedback

            self.steps += 1

    def _save_buffer_to_minari_file(self):
        for key in self.buffer.keys():
            self.buffer[key] = self.buffer[key][: self.steps + 2]

        episode_terminals = (
            self.buffer["terminations"] + self.buffer["truncations"]
            if self.args["include_timeout"]
            else None
        )

        md = MinariDataset(
            level_group=self._get_category(self._get_level()),
            level_name=self._get_level(),
            dataset_name=name_dataset(self.args),
            algorithm_name=self.args["policy"],
            environment_name=self.args["env_name"],
            seed_used=self.args["seed"],
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

    def _generate_new_dataset(self):
        # create folder to store MinariDataset files
        fp = os.environ.get("MINARI_DATASETS_PATH") or os.path.join(
            os.path.expanduser("~"), ".minari", "datasets"
        )
        self.fp, self.num_shards = os.path.join(fp, name_dataset(self.args)), 0
        if os.path.exists(self.fp):
            shutil.rmtree(self.fp)
        os.makedirs(self.fp)

        # create and initialise environment
        self.env = gym.make(self.args["env_name"])
        partial_obs, _ = self.env.reset(seed=self.args["seed"])
        obs = get_minigrid_obs(
            self.env, partial_obs, self.args["fully_obs"], self.args["rgb_obs"]
        )["image"]

        self.state_dim = np.prod(obs.shape)

        # initialise buffer to store replay data
        self._clear_buffer(obs.shape)

        # feedback verifiers
        self.rule_feedback_verifier = RuleFeedback()
        self.task_feedback_verifier = TaskFeedback(self.env)

        for ep_idx in tqdm(range(self.args["num_episodes"])):
            # create another episode
            self._create_episode()

            # if buffer contains 1000 episodes or this is final episode, save data to file and clear buffer
            if ((ep_idx + 1) % EPS_PER_SHARD == 0) or ep_idx == self.args["num_episodes"] - 1:
                self._save_buffer_to_minari_file()
                self._clear_buffer(obs.shape)

        self.env.close()
        self._clear_buffer(obs.shape)

    def _generate_new_dataset_OLD(self):
        """
        Generate a new dataset for a given environment, seed and policy.

        Returns
        -------
        MinariDataset: the dataset object that was created.
        """

        env = gym.make(self.args["env_name"])
        partial_obs, _ = env.reset(seed=self.args["seed"])
        obs = get_minigrid_obs(env, partial_obs, self.args["fully_obs"], self.args["rgb_obs"])

        replay_buffer = {
            "missions": [""] * (env.max_steps * self.args["num_episodes"] + 1),
            "observations": np.array(
                [np.zeros_like(obs["image"])]
                * (env.max_steps * self.args["num_episodes"] + 1),
                dtype=np.uint8,
            ),
            "actions": np.array(
                [[0]] * (env.max_steps * self.args["num_episodes"] + 1),
                dtype=np.float32,
            ),
            "rewards": np.array(
                [[0]] * (env.max_steps * self.args["num_episodes"] + 1),
                dtype=np.float32,
            ),
            "feedback": ["TEST"] * (env.max_steps * self.args["num_episodes"] + 1),
            "terminations": np.array(
                [[0]] * (env.max_steps * self.args["num_episodes"] + 1), dtype=bool
            ),
            "truncations": np.array(
                [[0]] * (env.max_steps * self.args["num_episodes"] + 1), dtype=bool
            ),
        }

        total_steps = 0
        terminated, truncated = False, False
        for _ in tqdm(range(self.args["num_episodes"])):
            partial_obs, _ = env.reset(seed=self.args["seed"])

            # Storing mission for initial episode timestep t=0 (m_0)
            # (mission is the same for the whole episode)
            mission = partial_obs["mission"]
            replay_buffer["missions"][total_steps] = mission

            # Storing observation at initial episode timestep t=0 (o_0)
            obs = get_minigrid_obs(
                env, partial_obs, self.args["fully_obs"], self.args["rgb_obs"]
            )
            replay_buffer["observations"][total_steps] = obs["image"]

            # Storing initial values for rewards, terminations, truncations and feedback
            replay_buffer["rewards"][total_steps] = np.array(0)
            replay_buffer["feedback"][total_steps] = "No feedback available."
            replay_buffer["terminations"][total_steps] = np.array(terminated)
            replay_buffer["truncations"][total_steps] = np.array(truncated)

            rule_feedback_verifier = RuleFeedback()
            task_feedback_verifier = TaskFeedback(env)
            terminated, truncated = False, False
            while not (terminated or truncated):
                # Passing partial observation to policy (PPO) as agent was trained on this
                # following the original implementation
                action = self._policy(partial_obs)
                rule_feedback = rule_feedback_verifier.verify_feedback(env, action)
                partial_obs, reward, terminated, truncated, _ = env.step(action)

                # Storing action a_t taken after observing o_t
                replay_buffer["actions"][total_steps] = np.array(action)

                # Generating and storing feedback f_t+1 resulting from taking a_t at o_t
                # If no rule feedback is provided (no rules have been breached), then
                # we set the feedback to be the task feedback, otherwise we set it to be
                # the rule feedback
                # (note that there should always either be rule feedback or task success feedback
                # as task success and rule violations are mutually exclusive)
                if rule_feedback == "No feedback available.":
                    feedback = task_feedback_verifier.verify_feedback(env, action)
                else:
                    feedback = rule_feedback

                # Storing observation o_t+1, reward r_t+1, termination r_t+1, truncation r_t+1
                # resulting from taking a_t at o_t
                obs = get_minigrid_obs(
                    env, partial_obs, self.args["fully_obs"], self.args["rgb_obs"]
                )
                replay_buffer["observations"][total_steps + 1] = obs["image"]
                replay_buffer["missions"][total_steps + 1] = mission
                replay_buffer["rewards"][total_steps + 1] = np.array(reward)
                replay_buffer["terminations"][total_steps + 1] = np.array(terminated)
                replay_buffer["truncations"][total_steps + 1] = np.array(truncated)
                replay_buffer["feedback"][total_steps + 1] = feedback

                total_steps += 1

        env.close()

        for key in replay_buffer.keys():
            replay_buffer[key] = replay_buffer[key][: total_steps + 2]

        episode_terminals = (
            replay_buffer["terminations"] + replay_buffer["truncations"]
            if self.args["include_timeout"]
            else None
        )

        return MinariDataset(
            level_group=self._get_category(self._get_level()),
            level_name=self._get_level(),
            dataset_name=name_dataset(self.args),
            algorithm_name=self.args["policy"],
            environment_name=self.args["env_name"],
            seed_used=self.args["seed"],
            code_permalink="https://github.com/maxtaylordavies/feedback-DT/blob/master/src/_datasets.py",
            author="Sabrina McCallum",
            author_email="s2431177@ed.ac.uk",
            missions=replay_buffer["missions"],
            observations=replay_buffer["observations"],
            actions=replay_buffer["actions"],
            rewards=replay_buffer["rewards"],
            feedback=replay_buffer["feedback"],
            terminations=replay_buffer["terminations"],
            truncations=replay_buffer["truncations"],
            episode_terminals=episode_terminals,
        )

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
            start += np.random.randint(0, self.episode_lengths[ep_idx])
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
            missions=np.array([]),
            feedback=np.array([]),
            dataset_name="",
            algorithm_name="",
            environment_name="",
            environment_stack="",
            seed_used=0,
            code_permalink="",
            author="",
            author_email="",
            observations=states,
            actions=actions,
            rewards=rewards,
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
            missions=np.array([]),
            feedback=np.array([]),
            dataset_name=f"dqn_replay-{game}-{num_samples}",
            algorithm_name="",
            environment_name="",
            environment_stack="",
            seed_used=0,
            code_permalink="",
            author="",
            author_email="",
            observations=np.array(obs),
            actions=np.array(acts),
            rewards=np.array(rewards),
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
