import json
import os
import re

import numpy as np
import gymnasium as gym
from gymnasium.utils.serialize_spec_stack import serialise_spec_stack
from jsonc_parser.parser import JsoncParser
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, RGBImgPartialObsWrapper
from tqdm import tqdm

from src.dataset.custom_dataset import CustomDataset
from src.dataset.minari_storage import list_local_datasets, name_dataset
from src.utils.utils import log
from src.utils.argparsing import get_args


def get_dataset(args):
    dataset_name = name_dataset(args)
    print(f"Creating dataset {dataset_name}")

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


def get_level(args):
    temp_level = re.sub(r"([SRN]\d).*", r"", args["env_name"].split("-")[1])
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


def get_category(level):
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


def get_used_action_space(args):
    file_path = "env_metadata.jsonc"
    metadata = JsoncParser.parse_file(file_path)
    level = get_level(args)
    category = get_category(level)
    return metadata["levels"][category][level]["used_action_space"]


def ppo(observation):
    raise Exception("This policy has not been implemented yet")


def policy(args, observation):
    if args["policy"] == "random_used_action_space_only":
        return np.random.choice(get_used_action_space(args))
    elif args["policy"] == "online_ppo":
        return ppo(observation["image"])
    else:
        return np.random.randint(0, 6)
    # return np.random.randint(0, 6)


def generate_new_dataset(args):
    env = gym.make(args["env_name"])
    observation, _ = env.reset(seed=args["seed"])
    print(f"Max steps used for array size: {env.max_steps}")

    fully_obs_env = FullyObsWrapper(env)
    rgb_env = RGBImgPartialObsWrapper(env)
    rgb_fully_obs_env = RGBImgObsWrapper(env)

    partial_observation = observation
    full_observation = fully_obs_env.observation({})
    rgb_partial_observation = rgb_env.observation({})
    rgb_full_observation = rgb_fully_obs_env.observation({})

    if args["fully_obs"]:
        if args["rgb_obs"]:
            observation = rgb_full_observation
        else:
            observation = full_observation
    else:
        if args["rgb_obs"]:
            observation = rgb_partial_observation
        else:
            observation = partial_observation

    agent_position = env.agent_pos

    environment_stack = serialise_spec_stack(env.spec_stack)

    replay_buffer = {
        "missions": [],
        "direction_observations": np.array(
            [[0]] * env.max_steps * args["num_episodes"], dtype=np.int32
        ),
        "agent_positions": np.array(
            [np.zeros_like(agent_position)] * env.max_steps * args["num_episodes"],
            dtype=np.uint8,
        ),
        "oracle_views": np.array(
            [np.zeros_like(full_observation["image"])]
            * env.max_steps
            * args["num_episodes"],
            dtype=np.uint8,
        ),
        "observations": np.array(
            [np.zeros_like(observation["image"])]
            * env.max_steps
            * args["num_episodes"],
            dtype=np.uint8,
        ),
        "actions": np.array(
            [[0]] * env.max_steps * args["num_episodes"], dtype=np.float32
        ),
        "rewards": np.array(
            [[0]] * env.max_steps * args["num_episodes"], dtype=np.float32
        ),
        "terminations": np.array(
            [[0]] * env.max_steps * args["num_episodes"], dtype=bool
        ),
        "truncations": np.array(
            [[0]] * env.max_steps * args["num_episodes"], dtype=bool
        ),
    }

    total_steps = 0
    for episode in tqdm(range(args["num_episodes"])):
        episode_steps, terminated, truncated = 0, False, False
        partial_observation, _ = env.reset(seed=args["seed"])
        fully_obs_env = FullyObsWrapper(env)

        while not (terminated or truncated):
            if episode_steps == 0:
                replay_buffer["missions"].append(partial_observation["mission"])
            else:
                replay_buffer["missions"].append("")
            replay_buffer["direction_observations"][total_steps] = np.array(
                partial_observation["direction"]
            )
            replay_buffer["agent_positions"][total_steps] = np.array(env.agent_pos)

            fully_obs_env = FullyObsWrapper(env)
            full_observation = fully_obs_env.observation({})

            rgb_env = RGBImgPartialObsWrapper(env)
            rgb_partial_observation = rgb_env.observation({})

            rgb_fully_obs_env = RGBImgObsWrapper(env)
            rgb_full_observation = rgb_fully_obs_env.observation({})

            replay_buffer["oracle_views"][total_steps] = np.array(
                full_observation["image"]
            )

            if args["fully_obs"]:
                if args["rgb_obs"]:
                    observation = rgb_full_observation
                else:
                    observation = full_observation
            else:
                if args["rgb_obs"]:
                    observation = rgb_partial_observation
                else:
                    observation = partial_observation

            replay_buffer["observations"][total_steps] = np.array(observation["image"])

            action = policy(args, observation)
            partial_observation, reward, terminated, truncated, _ = env.step(action)
            replay_buffer["actions"][total_steps] = np.array(action)
            replay_buffer["rewards"][total_steps] = np.array(reward)
            replay_buffer["terminations"][total_steps] = np.array(terminated)
            replay_buffer["truncations"][total_steps] = np.array(truncated)

            total_steps += 1
            episode_steps += 1

    env.close()

    for key in replay_buffer.keys():
        replay_buffer[key] = replay_buffer[key][:total_steps]

    episode_terminals = (
        replay_buffer["terminations"] + replay_buffer["truncations"]
        if args["include_timeout"]
        else None
    )

    return CustomDataset(
        level_group=get_category(get_level(args)),
        level_name=get_level(args),
        missions=replay_buffer["missions"],
        direction_observations=replay_buffer["direction_observations"],
        agent_positions=replay_buffer["agent_positions"],
        oracle_views=replay_buffer["oracle_views"],
        dataset_name=name_dataset(args),
        algorithm_name=args["policy"],
        environment_name=args["env_name"],
        environment_stack=json.dumps(environment_stack),
        seed_used=args["seed"],
        code_permalink="https://github.com/maxtaylordavies/feedback-DT/blob/master/src/_datasets.py",
        author="Sabrina McCallum",
        author_email="s2431177@ed.ac.uk",
        observations=replay_buffer["observations"],
        actions=replay_buffer["actions"],
        rewards=replay_buffer["rewards"],
        terminations=replay_buffer["terminations"],
        truncations=replay_buffer["truncations"],
        episode_terminals=episode_terminals,
    )


if __name__ == "__main__":
    args = get_args()
    dataset = get_dataset(args)
