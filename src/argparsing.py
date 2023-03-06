import argparse


def get_dataset_args():
    parser = argparse.ArgumentParser(description="Get dataset attributes")

    parser.add_argument(
        "--env-name",
        type=str,
        help="the name of the BabyAI environment; must be registered with gymnasium",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="the number of episodes to collect for the environment",
    )
    parser.add_argument(
        "--include-timeout",
        type=bool,
        default=True,
        help="whether to include episodes terminated by timeout / truncated",
    )

    return vars(parser.parse_args())


def delete_dataset_args():
    parser = argparse.ArgumentParser(
        description="Delete all or one specific dataset"
    )
    parser.add_argument(
        "--del-all",
        type=bool,
        help="whether to delete all local datasets",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default=None,
        help="the name of the BabyAI environment; must be registered with gymnasium",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="the number of episodes to collect for the environment",
    )
    parser.add_argument(
        "--include-timeout",
        type=bool,
        default=None,
        help="whether to include episodes terminated by timeout / truncated: 'incl-timeout' or 'excl-timeout'",
    )

    return vars(parser.parse_args())


def get_feedback_args():
    parser = argparse.ArgumentParser(
        description="Get feedback type and mode, as well as dataset attributes to identify the dataset"
    )
    parser.add_argument(
        "--type",
        type=str,
        help="the type of feedback to use: 'direction', 'distance', 'adjacency', or 'action'",
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="the feedback mode to use: 'simple' or 'verbose'",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        help="how often to provide feedback (every n-steps)",
    )
    parser.add_argument(
        "--freq-type",
        type=str,
        help="'exact' or 'poisson' - whether to provide feedback exactly every n-steps or use a poisson distribution",
    )
    
    parser.add_argument(
        "--env-name",
        type=str,
        help="the name of the BabyAI environment used for the dataset"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        help="the number of episodes in the dataset",
    )
    parser.add_argument(
        "--include-timeout",
        type=bool,
        help="whether the dataset includes episodes terminated by timeout / truncated",
    )

    return vars(parser.parse_args())


def get_training_args():
    parser = argparse.ArgumentParser(description="Decision transformer training")

    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        metavar="N",
        help="name of the run (default: dt-<date>)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="per-device batch size for training (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no_gpu", action="store_true", default=False, help="disables GPU training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="S",
        help="random seed (default: random number)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
        metavar="N",
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
        required=True,
        help="Path to the " "directory to write output to",
    )

    return vars(parser.parse_args())
