import os
from dataclasses import dataclass
import argparse

from datasets import load_dataset

from src.data import DecisionTransformerGymDataCollator
from src.dt import TrainableDT
from src.utils import log, setupDevices

os.environ[
    "WANDB_DISABLED"
] = "true"  # we diable weights and biases logging for this tutorial
dataset = load_dataset(
    "edbeeching/decision_transformer_gym_replay", "halfcheetah-expert-v2"
)


def construct_parser():
    # Training settings
    parser = argparse.ArgumentParser(description="Decision transformer training")
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-gpu", action="store_true", default=False, help="disables GPU training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="S",
        help="random seed (default: random number)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging " "training status",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="", help="path to pytorch checkpoint file"
    )
    return parser


def main(args):
    setupDevices(not args.no_gpu, args.seed)
    collator = DecisionTransformerGymDataCollator(dataset["train"])
    log("collator.n_traj", collator.n_traj)
    log("collator.state_mean", collator.state_mean)


if __name__ == "__main__":
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
