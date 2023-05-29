import os

import h5py
import numpy as np
from dopamine.replay_memory import circular_replay_buffer

from src.dataset.minari_dataset import MinariDataset


class CustomDataset(MinariDataset):
    def __init__(
        self,
        level_group,
        level_name,
        missions,
        direction_observations,
        agent_positions,
        oracle_views,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._level_group = level_group
        self._level_name = level_name
        self._missions = np.asarray(missions, dtype="S")
        self._direction_observations = np.asarray(
            direction_observations, dtype=np.int32
        ).reshape(-1)
        self._agent_positions = agent_positions
        self._oracle_views = oracle_views

    def save(self):
        """Saves dataset as HDF5.
        Args:
            fname (str): file path.
        """
        datasets_path = os.environ.get("MINARI_DATASETS_PATH") or os.path.join(
            os.path.expanduser("~"), ".minari", "datasets"
        )
        file_path = os.path.join(datasets_path, f"{self._dataset_name}.hdf5")

        os.makedirs(datasets_path, exist_ok=True)

        with h5py.File(file_path, "w") as f:
            f.create_dataset("level_group", data=self._level_group)
            f.create_dataset("level_name", data=self._level_name)
            f.create_dataset("missions", data=self._missions)
            f.create_dataset("direction_observations", data=self._direction_observations)
            f.create_dataset("agent_positions", data=self._agent_positions)
            f.create_dataset("oracle_views", data=self._oracle_views)
            f.create_dataset("dataset_name", data=self._dataset_name)
            f.create_dataset("algorithm_name", data=self._algorithm_name)
            f.create_dataset("environment_name", data=self._environment_name)
            f.create_dataset("environment_stack", data=str(self._environment_stack))
            f.create_dataset("seed_used", data=self._seed_used)
            f.create_dataset("code_permalink", data=str(self._code_permalink))
            f.create_dataset("author", data=str(self._author))
            f.create_dataset("author_email", data=str(self._author_email))
            f.create_dataset("observations", data=self._observations)
            f.create_dataset("actions", data=self._actions)
            f.create_dataset("rewards", data=self._rewards)
            f.create_dataset("terminations", data=self._terminations)
            f.create_dataset("truncations", data=self._truncations)
            f.create_dataset("episode_terminals", data=self._episode_terminals)
            f.create_dataset("discrete_action", data=self.discrete_action)
            f.create_dataset("version", data="1.0")
            f.flush()

    @property
    def level_group(self):
        return self._level_group

    @property
    def level_name(self):
        return self._level_name

    @property
    def missions(self):
        return self._missions

    @property
    def direction_observations(self):
        return self._direction_observations

    @property
    def agent_positions(self):
        return self._agent_positions

    @property
    def oracle_views(self):
        return self._oracle_views

    @classmethod
    def load(cls, dataset_name):
        """Loads dataset from HDF5.
        .. code-block:: python
            import numpy as np
            from minari.dataset import MinariDataset
            dataset = MinariDataset(np.random.random(10, 4),
                                 np.random.random(10, 2),
                                 np.random.random(10),
                                 np.random.randint(2, size=10))
            # save as HDF5
            dataset.dump('dataset.h5')
            # load from HDF5
            new_dataset = MinariDataset.load('dataset.h5')
        Args:
            fname (str): file .
        """
        datasets_path = os.environ.get("MINARI_DATASETS_PATH")
        if datasets_path is not None:
            file_path = os.path.join(datasets_path, f"{dataset_name}.hdf5")
        else:
            datasets_path = os.path.join(os.path.expanduser("~"), ".minari", "datasets")
            file_path = os.path.join(datasets_path, f"{dataset_name}.hdf5")

        with h5py.File(file_path, "r") as f:
            level_group = f["level_group"][()]
            level_name = f["level_name"][()]
            missions = f["missions"][()]
            direction_observations = f["direction_observations"][()]
            agent_positions = f["agent_positions"][()]
            oracle_views = f["oracle_views"][()]
            dataset_name = f["dataset_name"][()]
            algorithm_name = f["algorithm_name"][()]
            environment_name = f["environment_name"][()]
            environment_stack = f["environment_stack"][()]
            seed_used = f["seed_used"][()]
            code_permalink = f["code_permalink"][()]
            author = f["author"][()]
            author_email = f["author_email"][()]
            observations = f["observations"][()]
            actions = f["actions"][()]
            rewards = f["rewards"][()]
            terminations = f["terminations"][()]
            truncations = f["truncations"][()]
            discrete_action = f["discrete_action"][()]

            # for backward compatibility
            if "episode_terminals" in f:
                episode_terminals = f["episode_terminals"][()]
            else:
                episode_terminals = None

        dataset = cls(
            level_group=level_group,
            level_name=level_name,
            missions=missions,
            direction_observations=direction_observations,
            agent_positions=agent_positions,
            oracle_views=oracle_views,
            dataset_name=dataset_name,
            algorithm_name=algorithm_name,
            environment_name=environment_name,
            environment_stack=environment_stack,
            seed_used=seed_used,
            code_permalink=code_permalink,
            author=author,
            author_email=author_email,
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            episode_terminals=episode_terminals,
            discrete_action=discrete_action,
        )

        return dataset

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
            direction_observations=np.array([]),
            agent_positions=np.array([]),
            oracle_views=np.array([]),
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
            direction_observations=np.array([]),
            agent_positions=np.array([]),
            oracle_views=np.array([]),
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
