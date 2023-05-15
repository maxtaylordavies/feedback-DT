from datetime import datetime
import os

from transformers import DecisionTransformerConfig
import numpy as np
import wandb

from src.utils.argparsing import get_args
from src.collator import FeedbackDecisionTransformerDataCollator
from src.utils.utils import log, setup_devices, is_network_connection
from src.agent.fdt import FDTAgent
from src.trainer import AgentTrainer

from .generate_datasets import get_dataset
from .generate_feedback import get_feedback


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
    collator = FeedbackDecisionTransformerDataCollator(
        dataset,
        feedback=feedback,
        context_length=args["context_length"],
        randomise_starts=args["randomise_starts"],
    )

    # create the model
    config = DecisionTransformerConfig(
        state_dim=collator.state_dim, act_dim=collator.act_dim, max_length=64
    )
    agent = FDTAgent(config, use_feedback=args["use_feedback"])

    # train the model
    trainer = AgentTrainer(
        args=args,
        agent=agent,
        collator=collator,
        dataset=dataset,
    )
    trainer.train()

    # if using wandb, save args and finish run
    if args["wandb_mode"] != "disabled":
        wandb.config.update(args, allow_val_change=True)
        wandb.finish()


if __name__ == "__main__":
    args = get_args()
    log(f"parsed args: {args}")
    main(args)
