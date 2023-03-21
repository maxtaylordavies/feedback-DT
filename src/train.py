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
from src.collator import FeedbackDecisionTransformerDataCollator
from src._datasets import get_dataset
from src._feedback import get_feedback
from src.dt import FeedbackDT
from src.utils import log, setup_devices, is_network_connection
from src.evaluation import EvaluationCallback


def create_collator_and_model(args, dataset, feedback, device):
    # create the data collator
    collator = FeedbackDecisionTransformerDataCollator(
        dataset,
        feedback=feedback,
        context_length=args["context_length"],
        randomise_starts=args["randomise_starts"],
    )
    log(f"state_dim: {collator.state_dim}")
    log(f"act_dim: {collator.act_dim}")

    # create the model
    config = DecisionTransformerConfig(
        state_dim=collator.state_dim, act_dim=collator.act_dim, max_length=64
    )
    model = FeedbackDT(config, device)

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
        save_strategy="no",
    )

    # initialise the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[EvaluationCallback(user_args=args, collator=collator, gamma=1)],
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

    if args["seed"] == None:
        args["seed"] = np.random.randint(0, 2**32 - 1)
        log(f"seed not specified, using {args['seed']}")

    if args["policy"] == None:
        args["policy"] = lambda: np.random.randint(3)

    # setup compute devices
    device = setup_devices(args["seed"], not args["no_gpu"])

    # create or load training dataset
    dataset = get_dataset(args)
    print("np.max(dataset.actions):", np.max(dataset.actions))

    # create or load feedback if using feedback
    feedback = get_feedback(args, dataset) if args["use_feedback"] else None

    # create the data collator and model
    collator, model = create_collator_and_model(args, dataset, feedback, device)

    # train the model
    model = train_model(args, dataset, collator, model)

    # if using wandb, save args and finish run
    if args["wandb_mode"] != "disabled":
        wandb.config.update(args, allow_val_change=True)
        wandb.finish()


if __name__ == "__main__":
    args = get_args()
    log(f"parsed args: {args}")
    main(args)
