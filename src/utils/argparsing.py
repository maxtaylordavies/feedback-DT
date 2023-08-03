import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Decision transformer training")

    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="name of the run (default: current date and time)",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="BabyAI-GoToRedBallGrey-v0",
        help="the name of the environment config; must be registered with gymnasium",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="the number of episodes to collect for the environment",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="random",
    )
    parser.add_argument(
        "--include_timeout",
        type=bool,
        default=True,
        help="whether to include episodes terminated by timeout / truncated",
    )
    parser.add_argument(
        "--load_dataset_if_exists",
        type=bool,
        default=True,
        help="whether to load the dataset from local storage if it already exists",
    )
    parser.add_argument(
        "--fully_obs",
        type=bool,
        default=False,
        help="whether to use fully-observed environment",
    )
    parser.add_argument(
        "--rgb_obs",
        type=bool,
        default=True,
        help="whether to use rgb oberservations of the environment",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="per-device batch size for training (default: 64)",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=64,
        help="context length in timesteps (default: 64)",
    )
    parser.add_argument("--randomise_starts", type=bool, default=False)
    parser.add_argument(
        "--lr",
        type=float,
        default=5 * 1e-4,
        help="learning rate (default: 5 * 1e-4)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no_gpu", action="store_true", default=False, help="disables GPU training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed (default: 42)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="how many training steps between logging output (default: 10)",
    )
    parser.add_argument(
        "--record_video",
        type=bool,
        default=False,
        help="Whether to record videos of evaluation episodes",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="offline",
        help="wandb mode - can be online, offline, or disabled (default: offline)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="", help="path to pytorch checkpoint file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="/disk/scratch/feedback-DT/output",
        help="Path to the " "directory to write output to",
    )
    parser.add_argument(
        "--del_all",
        type=bool,
        default=False,
        help="Whether to delete all local datasets",
    )
    parser.add_argument(
        "--use_feedback",
        type=bool,
        default=True,
        help="whether to use feedback during training",
    )
    parser.add_argument(
        "--demo",
        type=str,
        default="from_default_hard",
        help="the type of demo to make, either from a predefined action sequence corresponding to an easy or a hard task, or from an action sequence corresponding to an episode of the actual training data for a given environment and seed.'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="demos",
        help="the directory to save output - such as demo videos - to.",
    )
    parser.add_argument(
        "--demo_episode",
        type=int,
        default=0,
        help="the index of the episode to make a demo video of.",
    )
    parser.add_argument(
        "--ppo_frames",
        type=int,
        default=10**7,
        help="the number of frames to train the PPO agent for.",
    )
    parser.add_argument(
        "--feedback_mode",
        type=str,
        default="all",
        help="which type of feedback to use during training; can be either 'all', 'rule_only', 'task_only', 'random', 'random_lorem_ipsum, or 'numerical_reward'.",
    )
    parser.add_argument(
        "--level",
        type=str,
        default="GoToRedBallGrey",
        help="the name of the level to train on.",
    )
    return vars(parser.parse_args())
