from datetime import datetime
import os

from datasets import load_dataset
from transformers import (
    DecisionTransformerConfig,
    Trainer,
    TrainingArguments,
)
import wandb

from src.argparsing import get_training_args
from src.data import DecisionTransformerGymDataCollator
from src.dt import TrainableDT
from src.utils import log, setup_devices, is_network_connection
from src.visualiser import visualise_trained_model


def load_data_and_create_model():
    # load the dataset
    dataset = load_dataset(
        "edbeeching/decision_transformer_gym_replay", "halfcheetah-expert-v2"
    )
    collator = DecisionTransformerGymDataCollator(dataset["train"])

    # create the model
    config = DecisionTransformerConfig(
        state_dim=collator.state_dim, act_dim=collator.act_dim
    )
    model = TrainableDT(config)

    return dataset, collator, model


def train_model(args, dataset, collator, model):
    # initialise the training args
    training_args = TrainingArguments(
        run_name=args["run_name"],
        output_dir=args["output"],
        report_to=None if args["disable_wandb"] else "wandb",
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
        train_dataset=dataset["train"],
        data_collator=collator,
    )

    # train the model
    log("Starting training...")
    trainer.train()
    log("Training complete :)")

    return model


def main(args):
    # do some setup
    if not args["run_name"]:
        args["run_name"] = f"dt-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        log(f"run_name not specified, using {args['run_name']}")
    
    if args["wandb_mode"] is not "disabled":
        os.environ["WANDB_PROJECT"] = "feedback-DT"
        os.environ["WANDB_LOG_MODEL"] = "true"
        os.environ["WANDB_WATCH"] = "false"

        if args["wandb_mode"] is "offline" or not is_network_connection():
            os.environ["WANDB_MODE"] = "dryrun"

    setup_devices(not args["no_gpu"], args["seed"])

    # load the data and create the model
    dataset, collator, model = load_data_and_create_model()

    # train the model
    model = train_model(args, dataset, collator, model)

    # visualise the trained model
    visualise_trained_model(args, collator, model, "HalfCheetah-v4")

    # finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    args = get_training_args()
    log(f"parsed args: {args}")
    main(args)
