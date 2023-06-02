import re

import gymnasium as gym
import numpy as np
from jsonc_parser.parser import JsoncParser
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, RGBImgPartialObsWrapper
from tqdm import tqdm

from src.dataset.custom_dataset import CustomDataset
from src.dataset.custom_feedback_verifier import RuleFeedback, TaskFeedback
from src.dataset.minari_storage import list_local_datasets, name_dataset
from src.utils.argparsing import get_args
from src.utils.utils import log


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


def get_observation(args, partial_observation, env):
    fully_obs_env = FullyObsWrapper(env)
    rgb_env = RGBImgPartialObsWrapper(env)
    rgb_fully_obs_env = RGBImgObsWrapper(env)

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

    return observation


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
        # Excluding the 'done' action (integer representation: 6), as by default, this is not used
        # to evaluate success for any of the tasks
        return np.random.randint(0, 6)


def generate_new_dataset(args):
    env = gym.make(args["env_name"])
    # DECIDE: SHOULD THIS BE DIFFERENT SEEDS? OTHERWISE THE ENV IS GOING TO BE IDENTICAL
    # IF SO, SHOULD WE CONTROL FOR SPECIFIC SEEDS FOR EVAL PURPOSES (ALSO SEE ISSUE WITH SEQ INSTRS)?
    partial_observation, _ = env.reset(seed=args["seed"])

    observation = get_observation(args, partial_observation, env)

    replay_buffer = {
        "missions": [""] * (env.max_steps * args["num_episodes"] + 1),
        "feedback": [""] * (env.max_steps * args["num_episodes"] + 1),
        "observations": np.array(
            [np.zeros_like(observation["image"])]
            * (env.max_steps * args["num_episodes"] + 1),
            dtype=np.uint8,
        ),
        "actions": np.array(
            [[0]] * (env.max_steps * args["num_episodes"] + 1), dtype=np.float32
        ),
        "rewards": np.array(
            [[0]] * (env.max_steps * args["num_episodes"] + 1), dtype=np.float32
        ),
        "terminations": np.array(
            [[0]] * (env.max_steps * args["num_episodes"] + 1), dtype=bool
        ),
        "truncations": np.array(
            [[0]] * (env.max_steps * args["num_episodes"] + 1), dtype=bool
        ),
    }

    total_steps = 0
    for episode in tqdm(range(args["num_episodes"])):
        terminated, truncated = False, False
        partial_observation, _ = env.reset(seed=args["seed"])
        # Mission is the same for the whole episode
        mission = partial_observation["mission"]
        # Storing mission for initial episode timestep t=0 (m_0)
        replay_buffer["missions"][total_steps] = mission
        # Storing observation at initial episode timestep t=0 (o_0)
        observation = get_observation(args, partial_observation, env)
        replay_buffer["observations"][total_steps] = observation["image"]
        # Storing initial values for rewards, terminations, truncations and feedback
        replay_buffer["rewards"][total_steps] = np.array(0)
        replay_buffer["terminations"][total_steps] = np.array(terminated)
        replay_buffer["truncations"][total_steps] = np.array(truncated)
        replay_buffer["feedback"][total_steps] = ""
        rule_feedback_verifier = RuleFeedback()
        task_feedback_verifier = TaskFeedback(env)

        while not (terminated or truncated):
            action = policy(args, observation)
            rule_feedback = rule_feedback_verifier.verify_feedback(env, action)
            partial_observation, reward, terminated, truncated, _ = env.step(action)

            # Storing action a_t taken after observing o_t
            replay_buffer["actions"][total_steps] = np.array(action)

            # Generating and storing feedback f_t+1 resulting from taking a_t at o_t
            # If no rule feedback is provided (no rules have been breached), then
            # we set the feedback to be the task feedback, otherwise we set it to be
            # the rule feedback
            # (note that there should always either be rule feedback or task success feedback
            # as task success and rule violations are mutually exclusive)
            if rule_feedback == "":
                feedback = task_feedback_verifier.verify_feedback(env, action)
            else:
                feedback = rule_feedback

            # Storing observation o_t+1, reward r_t+1, termination r_t+1, truncation r_t+1
            # resulting from taking a_t at o_t
            observation = get_observation(args, partial_observation, env)
            if not (terminated or truncated):
                replay_buffer["observations"][total_steps + 1] = observation["image"]
                replay_buffer["missions"][total_steps + 1] = mission
                replay_buffer["rewards"][total_steps + 1] = np.array(reward)
                replay_buffer["terminations"][total_steps + 1] = np.array(terminated)
                replay_buffer["truncations"][total_steps + 1] = np.array(truncated)
                replay_buffer["feedback"][total_steps + 1] = feedback

            total_steps += 1

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
        feedback=replay_buffer["feedback"],
        dataset_name=name_dataset(args),
        algorithm_name=args["policy"],
        environment_name=args["env_name"],
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
