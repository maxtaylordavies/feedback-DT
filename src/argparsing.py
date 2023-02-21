import argparse

def get_dataset_args():
    parser = argparse.ArgumentParser(description='Get admin token, mission and variant')
    
    parser.add_argument('--env-name', type=str,
                        help='the name of the environment; must be registered with gymnasium')
    parser.add_argument('--num-episodes', type=int, nargs=1, default=10,
                        help='the number of episodes to collect for the environment')
    parser.add_argument('--include-timeout', type=bool, nargs=1, default=True,
                        help='whether to include episodes terminated by timeout / truncated')
    
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
        default=10,
        metavar="N",
        help="how many batches to wait before logging " "training status",
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