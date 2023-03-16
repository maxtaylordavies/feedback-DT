import json
import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium.utils.serialize_spec_stack import serialise_spec_stack
from minari.storage.datasets_root_dir import get_file_path
from minigrid.core.world_object import Ball, Key, Box
from minigrid.wrappers import RGBImgObsWrapper

from argparsing import get_args
from custom_dataset import CustomDataset
from utils import log

basepath = os.path.dirname(os.path.dirname(os.path.abspath("")))
if not basepath in sys.path:
    sys.path.append(basepath)


def get_dataset(args):
    dataset_name = name_dataset(args)

    # optionally check if dataset already exists locally and load it
    if args["load_dataset_if_exists"] and dataset_name in list_local_datasets():
        log(f"Loading dataset {dataset_name} from local storage")
        return CustomDataset.load(dataset_name)

    # generate a new dataset, save locally and return
    else:
        dataset = generate_new_dataset(args)
        log(f"Created new dataset {dataset_name}, saving to local storage")
        dataset.save()
        return dataset


def list_local_datasets():
    datasets_path = get_file_path("").parent
    return [
        f[:-5]
        for f in os.listdir(datasets_path)
        if os.path.isfile(os.path.join(datasets_path, f))
    ]


def name_dataset(args):
    return f"{args['env_name']}_{args['num_episodes']}-eps_{'incl' if args['include_timeout'] else 'excl'}-timeout"


def generate_new_dataset(args):
    env = gym.make(args["env_name"])
    rgb_env = RGBImgObsWrapper(env)
    observation, _ = env.reset(seed=args["seed"])
    rgb_observation = rgb_env.observation({})
    agent_position = env.agent_pos

    # Get the environment specification stack for reproducibility
    environment_stack = serialise_spec_stack(env.spec_stack)

    replay_buffer = {
        "symbolic_observation": np.array([]),
        "goal_position": np.array([]),
        "agent_position": np.array([]),
        "direction_observation": np.array([]),
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
        "symbolic_observation": np.array(
            [np.zeros_like(observation["image"])]
            * env.max_steps
            * args["num_episodes"],
            dtype=np.uint8,
        ),
        "goal_position": np.array(
            [np.zeros_like(agent_position)] * env.max_steps * args["num_episodes"],
            dtype=np.uint8,
        ),
        "agent_position": np.array(
            [np.zeros_like(agent_position)] * env.max_steps * args["num_episodes"],
            dtype=np.uint8,
        ),
        "direction_observation": np.array(
            [[0]] * env.max_steps * args["num_episodes"], dtype=np.int32
        ),
        "episode": np.array(
            [[0]] * env.max_steps * args["num_episodes"], dtype=np.int32
        ),
        "observation": np.array(
            [np.zeros_like(rgb_observation["image"])]
            * env.max_steps
            * args["num_episodes"],
            dtype=np.uint8,
        ),
        "action": np.array(
            [[0]] * env.max_steps * args["num_episodes"], dtype=np.float32
        ),
        "reward": np.array(
            [[0]] * env.max_steps * args["num_episodes"], dtype=np.float32
        ),
        "terminated": np.array(
            [[0]] * env.max_steps * args["num_episodes"], dtype=bool
        ),
        "truncated": np.array([[0]] * env.max_steps * args["num_episodes"], dtype=bool),
    }

    total_steps = 0
    for episode in range(args["num_episodes"]):
        episode_step, terminated, truncated = 0, False, False
        obs, _ = env.reset(seed=args["seed"])
        rgb_env = RGBImgObsWrapper(env)
        # See DirectionObsWrapper() in Minigrid/minigrid/wrappers.py for original
        # implementation of the below
        goal_position_list = [
            x
            for x, y in enumerate(env.grid.grid)
            if y and y.type in obs["mission"] and y.color in obs["mission"]
        ]
        # TO-DO Adjust properly for cases with multiple goals
        # where we want to return all goal positions not just the first
        goal_position = (
            int(goal_position_list[0] / env.height),
            goal_position_list[0] % env.width,
        )
        while not (terminated or truncated):
            action = env.action_space.sample()  # User-defined policy function
            observation, reward, terminated, truncated, _ = env.step(action)
            rgb_observation = rgb_env.observation({})
            replay_buffer["symbolic_observation"][total_steps] = np.array(
                observation["image"]
            )
            replay_buffer["goal_position"][total_steps] = np.array(goal_position)
            replay_buffer["agent_position"][total_steps] = np.array(env.agent_pos)
            replay_buffer["direction_observation"][total_steps] = np.array(
                observation["direction"]
            )
            replay_buffer["episode"][total_steps] = np.array(episode)
            replay_buffer["observation"][total_steps] = np.array(
                rgb_observation["image"]
            )
            replay_buffer["action"][total_steps] = np.array(action)
            replay_buffer["reward"][total_steps] = np.array(reward)
            replay_buffer["terminated"][total_steps] = np.array(terminated)
            replay_buffer["truncated"][total_steps] = np.array(truncated)

            episode_step, total_steps = episode_step + 1, total_steps + 1

    env.close()

    # truncate the replay buffer to the actual number of steps taken
    for key in replay_buffer.keys():
        replay_buffer[key] = replay_buffer[key][:total_steps]

    if args["include_timeout"]:
        episode_terminals = replay_buffer["terminated"] + replay_buffer["truncated"]
    else:
        episode_terminals = None

    return CustomDataset(
        symbolic_observations=replay_buffer["symbolic_observation"],
        goal_positions=replay_buffer["goal_position"],
        agent_positions=replay_buffer["agent_position"],
        direction_observations=replay_buffer["direction_observation"],
        dataset_name=name_dataset(args),
        algorithm_name="random_policy",
        environment_name=args["env_name"],
        environment_stack=json.dumps(environment_stack),
        seed_used=args["seed"],
        code_permalink="https://github.com/maxtaylordavies/feedback-DT/blob/master/src/_datasets.py",
        author="Sabrina McCallum",
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
