import numpy as np

from src.argparsing import get_args
from src.collect_datasets import get_dataset
from jsonc_parser.parser import JsoncParser

args = get_args()
args["load_dataset_if_exists"] = False
args["seed"] = 0
args["policy"] = "random_used_action_space_only"
args["num_episodes"] = 10

file_path = "env_metadata.jsonc"
metadata = JsoncParser.parse_file(file_path)

for level_group in metadata["levels"].keys():
    for level in metadata["levels"][level_group].keys():
        for registered_config in metadata["levels"][level_group][level][
            "registered_configs"
        ].keys():
            print(registered_config)
            args["env_name"] = registered_config
            dataset = get_dataset(args)
            # print(len(dataset), np.max(dataset.actions))
