import os

import numpy as np
import torch
from transformers import DecisionTransformerConfig

from src.utils.utils import log
from src.utils.argparsing import get_args
from src.dataset.custom_dataset import CustomDataset
from src.collator import Collator
from src.agent.fdt import MinigridFDTAgent
from src.trainer import AgentTrainer

os.environ["WANDB_DISABLED"] = "true"
os.environ["ENV_METADATA_PATH"] = "/home/s2227283/projects/feedback-DT/env_metadata.jsonc"

args = get_args()

args["output"] = "/home/s2227283/projects/feedback-DT/data/output"
args["run_name"] = "july-1-test-1"
args["num_episodes"] = 100
args["load_dataset_if_exists"] = False
args["rgb_obs"] = True
args["seed"] = 0
args["policy"] = "random_used_action_space_only"
args["wandb_mode"] = "disabled"
args["report_to"] = "none"
args["epochs"] = 1
args["log_interval"] = 1

log("setting up devices")
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_str = f"{device.type}:{device.index}" if device.index else f"{device.type}"
    os.environ["CUDA_VISIBLE_DEVICES"] = device_str
    log(f"Using device: {torch.cuda.get_device_name()}")
else:
    device = torch.device("cpu")
    log("Using device: cpu")

log("creating dataset...")
dataset = CustomDataset.get_dataset(args)

log("creating collator...")
collator = Collator(custom_dataset=dataset, feedback=True, mission=True)
log(f"collator.observations.shape: {collator.observations.shape}")

log("creating agent...")
agent = MinigridFDTAgent(
    config=DecisionTransformerConfig(
        state_dim=collator.state_dim,
        act_dim=collator.act_dim,
        state_shape=(3, 56, 56),
        max_length=args["context_length"],
    )
)

log("creating trainer...")
trainer = AgentTrainer(
    agent=agent,
    collator=collator,
    dataset=dataset,
    args=args,
)

log("training agent...")
trainer.train()
