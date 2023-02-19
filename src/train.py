import os
from datetime import datetime
from dataclasses import dataclass
import argparse

from datasets import load_dataset
from transformers import (
    DecisionTransformerConfig,
    Trainer,
    TrainingArguments,
)

from src.data import DecisionTransformerGymDataCollator
from src.dt import TrainableDT
from src.utils import log, setup_devices

os.environ[
    "WANDB_DISABLED"
] = "true"  # we diable weights and biases logging for this tutorial


def construct_parser():
    # Training settings
    parser = argparse.ArgumentParser(description="Decision transformer training")
    parser.add_argument(
        "--run-name",
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
        "--batch-size",
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
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to the " "directory to write output to",
    )
    return parser


def main(args):
    if not args.run_name:
        args.run_name = f"dt-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    setup_devices(not args.no_gpu, args.seed)

    dataset = load_dataset(
        "edbeeching/decision_transformer_gym_replay", "halfcheetah-expert-v2"
    )

    collator = DecisionTransformerGymDataCollator(dataset["train"])
    log(f"collator.n_traj:, {collator.n_traj}")
    log(f"collator.state_mean: {collator.state_mean}")

    config = DecisionTransformerConfig(
        state_dim=collator.state_dim, act_dim=collator.act_dim
    )
    model = TrainableDT(config)

    training_args = TrainingArguments(
        run_name=args.run_name,
        output_dir=args.output,
        remove_unused_columns=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=1e-4,
        warmup_ratio=0.1,
        optim="adamw_torch",
        max_grad_norm=0.25,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=collator,
    )

    log("Starting training...")

    trainer.train()

    log("Training complete :)")


if __name__ == "__main__":
    parser = construct_parser()
    args = parser.parse_args()
    log(f"parsed args: {args}")
    main(args)
