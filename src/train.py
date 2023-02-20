import os
from datetime import datetime

from datasets import load_dataset
from transformers import (
    DecisionTransformerConfig,
    Trainer,
    TrainingArguments,
)

from src.argparsing import get_training_args
from src.data import DecisionTransformerGymDataCollator
from src.dt import TrainableDT
from src.utils import log, setup_devices

os.environ[
    "WANDB_DISABLED"
] = "true"  # we diable weights and biases logging for this tutorial


def main(args):
    if not args["run-name"]:
        args["run-name"] = f"dt-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        log(f"run-name not specified, using {args['run_name']}")

    setup_devices(not args["no-gpu"], args["seed"])

    dataset = load_dataset(
        "edbeeching/decision_transformer_gym_replay", "halfcheetah-expert-v2"
    )
    collator = DecisionTransformerGymDataCollator(dataset["train"])

    config = DecisionTransformerConfig(
        state_dim=collator.state_dim, act_dim=collator.act_dim
    )
    model = TrainableDT(config)

    training_args = TrainingArguments(
        run_name=args["run-name"],
        output_dir=args["output"],
        remove_unused_columns=False,
        num_train_epochs=args["epochs"],
        per_device_train_batch_size=args["batch-size"],
        learning_rate=args["lr"],
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
    args = get_training_args()
    log(f"parsed args: {args}")
    main(args)
