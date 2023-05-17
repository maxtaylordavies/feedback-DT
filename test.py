# import numpy as np

# from jsonc_parser.parser import JsoncParser

# from src.utils.argparsing import get_args
# from scripts.generate_datasets import get_dataset

# args = get_args()

# args["load_dataset_if_exists"] = False
# args["seed"] = 0
# args["policy"] = "random_used_action_space_only"
# args["num_episodes"] = 1

# metadata = JsoncParser.parse_file(
#     "/Users/sm2152/Documents/projects/feedback-DT/env_metadata.jsonc"
# )
# for group in metadata["levels"].keys():
#     for level in metadata["levels"][group].keys():
#         for config in metadata["levels"][group][level]["registered_configs"].keys():
#             if not "seq" in config.lower():
#                 args["env_name"] = config
#                 get_dataset(args)


ls = [""] * (5 * 2 + 1)
print(ls)
print(len(ls))
