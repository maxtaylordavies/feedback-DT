from src.dataset.custom_dataset import CustomDataset
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

if __name__ == "__main__":
    args = get_args()
    args["demo"] = "from_default_easy"
    if args["demo"] == "from_default_hard":
        config = DEFAULT_HARD_CONFIG
        seed = DEFAULT_HARD_SEED
        actions = DEFAULT_HARD_ACTIONS
    elif args["demo"] == "from_default_easy":
        config = DEFAULT_EASY_CONFIG
        seed = DEFAULT_EASY_SEED
        actions = DEFAULT_EASY_ACTIONS
    else:
        dataset = CustomDataset(args).get_dataset()
        config = dataset.environment_name
        seed = dataset.seed_used
        actions = dataset.episodes[args["demo_episode"]].actions
    demo = DemoVideo(config, seed, actions, args["demo"], args["output_dir"])
    demo.make_demo_video()
