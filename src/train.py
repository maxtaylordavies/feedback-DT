from datetime import datetime
import os

from transformers import (
    DecisionTransformerConfig,
    Trainer,
    TrainingArguments,
)
import numpy as np
import wandb

from src.argparsing import get_args
from src.collator import DecisionTransformerMinariDataCollator
from src._datasets import get_dataset
from src.dt import FeedbackDT
from src.utils import log, setup_devices, is_network_connection
from src.visualiser import visualise_trained_model


def create_collator_and_model(dataset):
    # create the data collator
    collator = DecisionTransformerMinariDataCollator(dataset)
    log(f"state_dim: {collator.state_dim}")
    log(f"act_dim: {collator.act_dim}")

    # create the model
    config = DecisionTransformerConfig(state_dim=collator.state_dim, act_dim=collator.act_dim)
    model = FeedbackDT(config)

    return collator, model


def train_model(args, dataset, collator, model):
    # initialise the training args
    training_args = TrainingArguments(
        run_name=args["run_name"],
        output_dir=args["output"],
        report_to=None if args["wandb_mode"] == "disabled" else "wandb",
        logging_steps=args["log_interval"],
        remove_unused_columns=False,
        num_train_epochs=args["epochs"],
        per_device_train_batch_size=args["batch_size"],
        learning_rate=args["lr"],
        weight_decay=1e-4,
        warmup_ratio=0.1,
        optim="adamw_torch",
        max_grad_norm=0.25,
    )

    # initialise the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    # train the model
    log("Starting training...")
    trainer.train()
    log("Training complete :)")

    return model


def main(args):
    # set some variables
    if not args["run_name"]:
        args["run_name"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log(f"run_name not specified, using {args['run_name']}")

    if args["wandb_mode"] != "disabled":
        os.environ["WANDB_PROJECT"] = "feedback-DT"
        os.environ["WANDB_LOG_MODEL"] = "false"
        os.environ["WANDB_WATCH"] = "false"
        os.environ["WANDB__SERVICE_WAIT"] = "300"

        if args["wandb_mode"] == "offline" or not is_network_connection():
            log("using wandb in offline mode")
            os.environ["WANDB_MODE"] = "dryrun"

    if not args["seed"]:
        args["seed"] = np.random.randint(0, 2**32 - 1)
        log(f"seed not specified, using {args['seed']}")

    # setup compute devices
    setup_devices(args["seed"], not args["no_gpu"])

    # create or load training dataset
    dataset = get_dataset(args)

    # create the data collator and model
    collator, model = create_collator_and_model(dataset)

    # train the model
    # model = train_model(args, dataset, collator, model)

    # visualise the trained model
    visualise_trained_model(args, collator, model, epochs_trained=0)

    # if using wandb, save args and finish run
    if args["wandb_mode"] != "disabled":
        wandb.config.update(args)
        wandb.finish()


if __name__ == "__main__":
    args = get_args()
    log(f"parsed args: {args}")
    main(args)
