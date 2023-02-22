import json
import os

import gymnasium as gym
from gymnasium.utils.serialize_spec_stack import serialise_spec_stack
import minari
from minari.dataset import MinariDataset
from minari.storage.datasets_root_dir import get_file_path
import minigrid 
import numpy as np

from src.argparsing import get_dataset_args

def generate_dataset(env_name: str, num_episodes: int, include_timeout: bool, seed=42):
    env = gym.make(env_name)
    # TODO Consider whether there is any advantage in using the RGBImgPartialObsWrapper()

    observation, _ = env.reset(seed=seed)

    # Get the environment specification stack for reproducibility
    environment_stack = serialise_spec_stack(
        env.spec_stack
    ) 

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
        "episode": np.array(
            [[0]] * env.max_steps * num_episodes, dtype=np.int32
        ),
        # Adjusted this for BabyAI image observation shape (7, 7, 3)
        "observation": np.array(
            [np.zeros_like(observation['image'])] * env.max_steps * num_episodes,
            dtype=np.uint8,
        ),
        # Adjusted this for discrete actions space
        "action": np.array(
            [[0]] * env.max_steps * num_episodes, dtype=np.float32
        ),
        "reward": np.array(
            [[0]] * env.max_steps * num_episodes, dtype=np.float32
        ),
        "terminated": np.array(
            [[0]] * env.max_steps * num_episodes, dtype=bool
        ),
        "truncated": np.array(
            [[0]] * env.max_steps * num_episodes, dtype=bool
        ),
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
            replay_buffer["observation"][total_steps] = np.array(observation['image'])
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
    
    dataset_name = name_dataset(env_name, num_episodes, include_timeout)
    
    dataset = MinariDataset(
        dataset_name=dataset_name,
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
        episode_terminals=episode_terminals
    )
    
    return dataset

def name_dataset(env_name: str, num_episodes: int, include_timeout: bool):
    if include_timeout:
        dataset_version_suffix = "incl-timeout"
    else:
        dataset_version_suffix = "excl-timeout"
        
    dataset_name = f'{env_name}_{num_episodes}-eps_{dataset_version_suffix}'
    
    return dataset_name

def get_local_datasets():
    datasets_path = get_file_path("").parent
    datasets = [
        f[:-5]
        for f in os.listdir(datasets_path)
        if os.path.isfile(os.path.join(datasets_path, f))
    ]
    return datasets


if __name__ == "__main__":
    args = get_dataset_args()
    env_name = args["env-name"]
    num_episodes = args["num-episodes"]
    include_timeout = args["include-timeout"]
    
    dataset_name = name_dataset(env_name, num_episodes, include_timeout)
    local_datasets = get_local_datasets()
    
    if dataset_name in local_datasets:
        dataset = minari.load_dataset(dataset_name)
        print(f"Success! Loaded dataset {dataset_name}")
    else:
        dataset = generate_dataset(env_name, num_episodes, include_timeout)
        print(f"Success! Generated dataset {dataset_name}")

        dataset.save()
        print(f"Success! Saved dataset {dataset_name}")

    
    


