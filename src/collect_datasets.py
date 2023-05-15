import json
import os
import re

import gymnasium as gym
import numpy as np
from custom_dataset import CustomDataset
from gymnasium.utils.serialize_spec_stack import serialise_spec_stack
from jsonc_parser.parser import JsoncParser
from minari_storage import get_file_path
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, RGBImgPartialObsWrapper
from argparsing import get_args
from utils import log
from tqdm import tqdm


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


def list_local_datasets():
    datasets_path = get_file_path("").parent
    return [
        f[:-5]
        for f in os.listdir(datasets_path)
        if os.path.isfile(os.path.join(datasets_path, f))
    ]


def name_dataset(args):
    return f"{args['env_name']}_{args['num_episodes']}-eps_{args['policy'].replace('_', '-')}_{'incl' if args['include_timeout'] else 'excl'}-timeout"


def get_level(args):
    return re.sub(r"([SRN]\d).*", r"", args["env_name"].split("-")[1])


def get_category(level):
    if level.startswith("GoTo"):
        return "GoTo"
    elif level.startswith("Open"):
        return "Open"
    elif level.startswith("Pickup"):
        return "Pickup"
    elif level.startswith("PutNext"):
        return "PutNext"
    elif level.startswith("Unlock"):
        return "Unlock"
    elif level.startswith("Synth"):
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
        "mission": [],
        "direction_observation": np.array(
            [[0]] * env.max_steps * args["num_episodes"], dtype=np.int32
        ),
        "goal_position": np.array(
            [np.zeros_like(agent_position)] * env.max_steps * args["num_episodes"],
            dtype=np.uint8,
        ),
        "agent_position": np.array(
            [np.zeros_like(agent_position)] * env.max_steps * args["num_episodes"],
            dtype=np.uint8,
        ),
        "oracle_view": np.array(
            [np.zeros_like(full_observation["image"])]
            * env.max_steps
            * args["num_episodes"],
            dtype=np.uint8,
        ),
        "observation": np.array(
            [np.zeros_like(observation["image"])]
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
    for episode in tqdm(range(args["num_episodes"])):
        terminated, truncated = False, False
        partial_observation, _ = env.reset(seed=args["seed"])
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
            replay_buffer["mission"].append(partial_observation["mission"])
            replay_buffer["direction_observation"][total_steps] = np.array(
                partial_observation["direction"]
            )
            replay_buffer["goal_position"][total_steps] = np.array(goal_position)
            replay_buffer["agent_position"][total_steps] = np.array(env.agent_pos)

            fully_obs_env = FullyObsWrapper(env)
            full_observation = fully_obs_env.observation({})

            rgb_env = RGBImgPartialObsWrapper(env)
            rgb_partial_observation = rgb_env.observation({})

            rgb_fully_obs_env = RGBImgObsWrapper(env)
            rgb_full_observation = rgb_fully_obs_env.observation({})

            replay_buffer["oracle_view"][total_steps] = np.array(
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

            replay_buffer["observation"][total_steps] = np.array(observation["image"])

            action = policy(args, observation)
            partial_observation, reward, terminated, truncated, _ = env.step(action)
            replay_buffer["action"][total_steps] = np.array(action)
            replay_buffer["reward"][total_steps] = np.array(reward)
            replay_buffer["terminated"][total_steps] = np.array(terminated)
            replay_buffer["truncated"][total_steps] = np.array(truncated)

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
        level_group=get_category(get_level(args)),
        level_name=get_level(args),
        mission=replay_buffer["mission"],
        direction_observations=replay_buffer["direction_observation"],
        goal_positions=replay_buffer["goal_position"],
        agent_positions=replay_buffer["agent_position"],
        oracle_view=replay_buffer["oracle_view"],
        dataset_name=name_dataset(args),
        algorithm_name=args["policy"],
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
