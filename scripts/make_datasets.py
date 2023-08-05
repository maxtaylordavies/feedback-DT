import numpy as np

from src.utils.argparsing import get_args
from src.dataset.custom_dataset import CustomDataset
from src.dataset.seeds import LEVELS_CONFIGS

args = get_args()

for policy in ("random", "ppo"):
    if policy == "ppo":
        continue
    args["policy"] = policy
    for feedback_mode in [
        "all",
        "rule_only",
        "task_only",
        "random",
        "random_lorem_ipsum",
        "numerical_reward",
    ]:
        args["feedback_mode"] = feedback_mode
        for level in LEVELS_CONFIGS["original_tasks"]:
            args["level"] = level
        dataset = CustomDataset(args)
        data = dataset.get_dataset(args)
