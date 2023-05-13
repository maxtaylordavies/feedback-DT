from jsonc_parser.parser import JsoncParser
from src.argparsing import get_args

import numpy as np

args = get_args()

print(args)


def get_used_action_space(args):
    file_path = "env_metadata.jsonc"
    metadata = JsoncParser.parse_file(file_path)
    level = args["env_name"].split("-")[1]
    return metadata[level]["used_action_space"]


def ppo(observation):
    raise Exception("This policy has not been implemented yet")


def pi(observation):
    if args["policy"] == "used_action_space_only":
        return np.random.choice(1, get_used_action_space(args))
    elif args["policy"] == "online":
        return ppo(observation)
    else:
        return np.random.randint(0, 6)


if __name__ == "__main__":
    print(pi(1))
