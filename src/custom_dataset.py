import os

import h5py
import numpy as np
from minari.dataset import MinariDataset
from minari.logger import LOG


class CustomDataset(MinariDataset):
    def __init__(
        self,
        symbolic_observations,
        goal_positions,
        agent_positions,
        direction_observations,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._symbolic_observations = symbolic_observations
        self._goal_positions = goal_positions
        self._agent_positions = agent_positions
        self._direction_observations = np.asarray(
            direction_observations, dtype=np.int32
        ).reshape(-1)

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
            f.create_dataset("symbolic_observations", data=self._symbolic_observations)
            f.create_dataset("goal_positions", data=self._goal_positions)
            f.create_dataset("agent_positions", data=self._agent_positions)
            f.create_dataset("direction_observations", data=self._direction_observations)
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
            symbolic_observations = f["symbolic_observations"][()]
            goal_positions = f["goal_positions"][()]
            agent_positions = f["agent_positions"][()]
            direction_observations = f["direction_observations"][()]
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

            if "version" not in f:
                LOG.warning("The dataset structure might be incompatible.")

        dataset = cls(
            symbolic_observations=symbolic_observations,
            goal_positions=goal_positions,
            agent_positions=agent_positions,
            direction_observations=direction_observations,
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

    @property
    def symbolic_observations(self):
        return self._symbolic_observations

    @property
    def goal_positions(self):
        return self._goal_positions

    @property
    def agent_positions(self):
        return self._agent_positions

    @property
    def direction_observations(self):
        return self._direction_observations
