import json
import os

import gymnasium as gym
from gymnasium.utils.serialize_spec_stack import serialise_spec_stack
import minari
from minari.dataset import MinariDataset
from minari.storage.datasets_root_dir import get_file_path
import numpy as np

from src.argparsing import get_args
from src.utils import log


def get_dataset(args):
    dataset_name = name_dataset(args)

    # optionally check if dataset already exists locally and load it
    if args["load_dataset_if_exists"] and dataset_name in list_local_datasets():
        log(f"Loading dataset {dataset_name} from local storage")
        return minari.load_dataset(dataset_name)

    # generate a new dataset
    return generate_new_dataset(
        env_name=args["env_name"],
        num_episodes=args["num_episodes"],
        include_timeout=args["include_timeout"],
        seed=args["seed"],
    )


def list_local_datasets():
    datasets_path = get_file_path("").parent
    return [
        f[:-5]
        for f in os.listdir(datasets_path)
        if os.path.isfile(os.path.join(datasets_path, f))
    ]


def name_dataset(args):
    return f"{args['env_name']}_{args['num_episodes']}-eps_{'incl' if args['include_timeout'] else 'excl'}-timeout"


def generate_new_dataset(env_name: str, num_episodes: int, include_timeout: bool, seed: int):
    env = gym.make(env_name)
    # TODO Consider whether there is any advantage in using the RGBImgPartialObsWrapper()

    observation, _ = env.reset(seed=seed)

    # Get the environment specification stack for reproducibility
    environment_stack = serialise_spec_stack(env.spec_stack)

    replay_buffer = {
        "episode": np.array([]),
        "observation": np.array([]),
        "action": np.array([]),
        "reward": np.array([]),
        "terminated": np.array([]),
        "truncated": np.array([]),
    }

    # Using env.max_steps instead of env.spec.max_episode_steps, as the latter was not defined
    # upon registering BabyAI envs as Gymnasium envs (so that env.spec.mex_episode_steps = None)
    replay_buffer = {
        "episode": np.array([[0]] * env.max_steps * num_episodes, dtype=np.int32),
        # Adjusted this for BabyAI image observation shape (7, 7, 3)
        "observation": np.array(
            [np.zeros_like(observation["image"])] * env.max_steps * num_episodes,
            dtype=np.uint8,
        ),
        # Adjusted this for discrete actions space
        "action": np.array([[0]] * env.max_steps * num_episodes, dtype=np.float32),
        "reward": np.array([[0]] * env.max_steps * num_episodes, dtype=np.float32),
        "terminated": np.array([[0]] * env.max_steps * num_episodes, dtype=bool),
        "truncated": np.array([[0]] * env.max_steps * num_episodes, dtype=bool),
    }

    total_steps = 0
    for episode in range(num_episodes):
        episode_step = 0
        env.reset(seed=42)
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = env.action_space.sample()  # User-defined policy function
            observation, reward, terminated, truncated, _ = env.step(action)

            replay_buffer["episode"][total_steps] = np.array(episode)
            replay_buffer["observation"][total_steps] = np.array(observation["image"])
            replay_buffer["action"][total_steps] = np.array(action)
            replay_buffer["reward"][total_steps] = np.array(reward)
            replay_buffer["terminated"][total_steps] = np.array(terminated)
            replay_buffer["truncated"][total_steps] = np.array(truncated)

            episode_step += 1
            total_steps += 1

    env.close()

    replay_buffer["episode"] = replay_buffer["episode"][:total_steps]
    replay_buffer["observation"] = replay_buffer["observation"][:total_steps]
    replay_buffer["action"] = replay_buffer["action"][:total_steps]
    replay_buffer["reward"] = replay_buffer["reward"][:total_steps]
    replay_buffer["terminated"] = replay_buffer["terminated"][:total_steps]
    replay_buffer["truncated"] = replay_buffer["truncated"][:total_steps]

    if include_timeout:
        episode_terminals = replay_buffer["terminated"] + replay_buffer["truncated"]
    else:
        episode_terminals = None

    return MinariDataset(
        dataset_name=name_dataset(env_name, num_episodes, include_timeout),
        algorithm_name="random_policy",
        environment_name=env_name,
        environment_stack=json.dumps(environment_stack),
        seed_used=42,
        code_permalink=None,
        author="SabrinaMcCallum",
        author_email="s2431177@ed.ac.uk",
        observations=replay_buffer["observation"],
        actions=replay_buffer["action"],
        rewards=replay_buffer["reward"],
        terminations=replay_buffer["terminated"],
        truncations=replay_buffer["truncated"],
        episode_terminals=episode_terminals,
    )


if __name__ == "__main__":
    args = get_args()
    dataset = get_dataset(args)
    dataset.save()
    print(f"Success! Saved dataset {dataset.dataset_name}")
