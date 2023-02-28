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
        help="the name of the environment; must be registered with gymnasium",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="the number of episodes to collect for the environment",
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
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate (default: 1e-4)",
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
        default=None,
        help="random seed (default: random number)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
        help="how many batches to wait before logging training status (default: 1)",
    )
    parser.add_argument(
        "--visualise-interval",
        type=str,
        default="end",
        help="interval at which to visualise model's performance in the environment (default: end)",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        help="wandb mode - can be online, offline, or disabled (default: online)",
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

    return vars(parser.parse_args())
