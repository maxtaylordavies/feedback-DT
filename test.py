# from jsonc_parser.parser import JsoncParser

# from scripts.generate_datasets import get_dataset
# from src.utils.argparsing import get_args

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


import gymnasium as gym
import matplotlib.pyplot as plt
from minigrid.wrappers import RGBImgObsWrapper

env = gym.make("BabyAI-MiniBossLevel-v0")
env.reset(seed=16)


rgb_env = RGBImgObsWrapper(env)
for action in [1, 1, 3, 1, 5]:
    rgb_obs, _ = rgb_env.step(action)
    plt.imshow(rgb_obs["image"])
    plt.show()
