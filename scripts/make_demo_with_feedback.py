from src.utils.argparsing import get_args
from src.utils.demos import (
    DEFAULT_EASY_ACTIONS,
    DEFAULT_EASY_CONFIG,
    DEFAULT_EASY_SEED,
    DEFAULT_HARD_ACTIONS,
    DEFAULT_HARD_CONFIG,
    DEFAULT_HARD_SEED,
    DemoVideo,
)
from generate_datasets import get_dataset

if __name__ == "__main__":
    args = get_args()
    args["load_dataset_if_exists"] = False
    args["demo"] = "custom"
    args["demo_episode"] = 0
    if args["demo"] == "from_default_hard":
        config = DEFAULT_HARD_CONFIG
        seed = DEFAULT_HARD_SEED
        actions = DEFAULT_HARD_ACTIONS
    elif args["demo"] == "from_default_easy":
        config = DEFAULT_EASY_CONFIG
        seed = DEFAULT_EASY_SEED
        actions = DEFAULT_EASY_ACTIONS
    else:
        dataset = get_dataset(args)
        config = dataset.environment_name
        seed = dataset.seed_used
        actions = dataset.episodes[args["demo_episode"]].actions
    # demo = DemoVideo(config, seed, actions, args["demo"], args["output_dir"])
    # demo.make_demo_video()
