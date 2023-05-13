import json
from jsonc_parser.parser import JsoncParser
import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium.utils.serialize_spec_stack import serialise_spec_stack
from minari.storage.datasets_root_dir import get_file_path
from minigrid.wrappers import RGBImgPartialObsWrapper, FullyObsWrapper, RGBImgObsWrapper
from tqdm import tqdm

from src.argparsing import get_args
from src.custom_dataset import CustomDataset
from src.utils import log


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


def get_used_action_space(args):
    file_path = "env_metadata.jsonc"
    metadata = JsoncParser.parse_file(file_path)
    return metadata[args["env_name"]]["used_action_space"]


def pi(observation):
    if args["policy"]:
        if "online" in args["policy"]:
            return args["policy"](observation)
        else:
            return np.random.choice(1, get_used_action_space(args))
    else:
        return np.random.randint(0, 6)


def generate_new_dataset(args):
    env = gym.make(args["env_name"])
    env.reset(seed=args["seed"])
    fully_obs_env = FullyObsWrapper(env)
    rgb_env = RGBImgPartialObsWrapper(env)
    rgb_fully_obs_env = RGBImgObsWrapper(env)

    partial_observation = env.observation({})
    full_observation = fully_obs_env.observation({})
    rgb_partial_observation = rgb_env.observation({})
    rgb_full_observation = rgb_fully_obs_env.observation({})

    agent_position = env.agent_pos

    # Get the environment specification stack for reproducibility
    environment_stack = serialise_spec_stack(env.spec_stack)

    # Using env.max_steps instead of env.spec.max_episode_steps, as the latter was not defined
    # upon registering BabyAI envs as Gymnasium envs (so that env.spec.mex_episode_steps = None)
    replay_buffer = {
        "reward": np.array(
            [[0]] * env.max_steps * args["num_episodes"], dtype=np.float32
        ),
        "partial_observation": np.array(
            [np.zeros_like(partial_observation["image"])]
            * env.max_steps
            * args["num_episodes"],
            dtype=np.uint8,
        ),
        "full_observation": np.array(
            [np.zeros_like(full_observation["image"])]
            * env.max_steps
            * args["num_episodes"],
            dtype=np.uint8,
        ),
        "rgb_partial_observation": np.array(
            [np.zeros_like(rgb_partial_observation["image"])]
            * env.max_steps
            * args["num_episodes"],
            dtype=np.uint8,
        ),
        "rgb_full_observation": np.array(
            [np.zeros_like(rgb_full_observation["image"])]
            * env.max_steps
            * args["num_episodes"],
            dtype=np.uint8,
        ),
        "direction_observation": np.array(
            [[0]] * env.max_steps * args["num_episodes"], dtype=np.int32
        ),
        "mission": np.array([[0]] * env.max_steps * args["num_episodes"], dtype=str),
        "goal_position": np.array(
            [np.zeros_like(agent_position)] * env.max_steps * args["num_episodes"],
            dtype=np.uint8,
        ),
        "agent_position": np.array(
            [np.zeros_like(agent_position)] * env.max_steps * args["num_episodes"],
            dtype=np.uint8,
        ),
        "terminated": np.array(
            [[0]] * env.max_steps * args["num_episodes"], dtype=bool
        ),
        "truncated": np.array([[0]] * env.max_steps * args["num_episodes"], dtype=bool),
        "action": np.array(
            [[0]] * env.max_steps * args["num_episodes"], dtype=np.float32
        ),
    }

    total_steps = 0
    for episode in tqdm(range(args["num_episodes"])):
        terminated, truncated = False, False
        partial_observation, reward, _ = env.reset(seed=args["seed"])

        fully_obs_env = FullyObsWrapper(env)
        goal_position_list = [
            x
            for x, y in enumerate(fully_obs_env.grid.grid)
            if y
            and y.type in partial_observation["mission"]
            and y.color in partial_observation["mission"]
        ]

        # For cases with multiple goals, we want to return a random goal's position
        np.random.shuffle(goal_position_list)
        goal_position = (
            goal_position_list[0] % env.width,
            int(goal_position_list[0] / env.height),
        )

        while not (terminated or truncated):
            replay_buffer["reward"][total_steps] = np.array(reward)
            replay_buffer["partial_observation"][total_steps] = np.array(
                partial_observation["image"]
            )

            fully_obs_env = FullyObsWrapper(env)
            full_observation = fully_obs_env.observation({})
            replay_buffer["full_observation"][total_steps] = np.array(
                full_observation["image"]
            )

            rgb_env = RGBImgPartialObsWrapper(env)
            rgb_partial_observation = rgb_env.observation({})
            replay_buffer["rgb_partial_observation"][total_steps] = np.array(
                rgb_partial_observation["image"]
            )

            rgb_fully_obs_env = RGBImgObsWrapper(env)
            rgb_full_observation = rgb_fully_obs_env.observation({})
            replay_buffer["rgb_full_observation"][total_steps] = np.array(
                rgb_full_observation["image"]
            )

            replay_buffer["direction_observation"][total_steps] = np.array(
                partial_observation["direction"]
            )
            replay_buffer["mission"][total_steps] = np.array(
                partial_observation["mission"]
            )
            replay_buffer["goal_position"][total_steps] = np.array(goal_position)
            replay_buffer["agent_position"][total_steps] = np.array(env.agent_pos)
            replay_buffer["terminated"][total_steps] = np.array(terminated)
            replay_buffer["truncated"][total_steps] = np.array(truncated)

            action = pi(partial_observation)
            replay_buffer["action"][total_steps] = np.array(action)

            partial_observation, reward, terminated, truncated, _ = env.step(action)

            total_steps += 1

    env.close()

    for key in replay_buffer.keys():
        replay_buffer[key] = replay_buffer[key][:total_steps]

    episode_terminals = (
        replay_buffer["terminated"] + replay_buffer["truncated"]
        if args["include_timeout"]
        else None
    )

    return CustomDataset(
        rewards=replay_buffer["reward"],
        partial_observations=replay_buffer["partial_observation"],
        full_observations=replay_buffer["full_observation"],
        rgb_partial_observations=replay_buffer["rgb_partial_observation"],
        rgb_full_observations=replay_buffer["rgb_full_observation"],
        direction_observations=replay_buffer["direction_observation"],
        mission=replay_buffer["mission"],
        goal_positions=replay_buffer["goal_position"],
        agent_positions=replay_buffer["agent_position"],
        actions=replay_buffer["action"],
        terminations=replay_buffer["terminated"],
        truncations=replay_buffer["truncated"],
        episode_terminals=episode_terminals,
        dataset_name=name_dataset(args),
        algorithm_name=pi.name,
        environment_name=args["env_name"],
        environment_stack=json.dumps(environment_stack),
        seed_used=args["seed"],
        code_permalink="https://github.com/maxtaylordavies/feedback-DT/blob/master/src/_datasets.py",
        author="Sabrina McCallum",
        author_email="s2431177@ed.ac.uk",
    )


if __name__ == "__main__":
    args = get_args()
    dataset = get_dataset(args)
