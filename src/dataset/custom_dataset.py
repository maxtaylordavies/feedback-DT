import os

import h5py
import numpy as np

from src.dataset.minari_dataset import MinariDataset


class CustomDataset(MinariDataset):
    def __init__(
        self,
        level_group,
        level_name,
        missions,
        feedback
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._level_group = level_group
        self._level_name = level_name
        self._missions = np.asarray(missions, dtype="S")
        self._feedback = np.asarray(feedback, dtype="S")

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
            f.create_dataset("missions", data=self._feedback)
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
            level_group = f["level_group"][()]
            level_name = f["level_name"][()]
            missions = f["missions"][()]
            feedback = f["feedback"][()]
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
            feedback=feedback,
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
    def level_group(self):
        return self._level_group

    @property
    def level_name(self):
        return self._level_name

    @property
    def missions(self):
        return self._missions

    @property
    def missions(self):
        return self._missions