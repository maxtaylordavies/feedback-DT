import os

import torch
from transformers import DecisionTransformerConfig

from src.utils.utils import log
from src.utils.argparsing import get_args
from src.dataset.custom_dataset import CustomDataset
from src.collator import Collator, RoundRobinCollator, CurriculumCollator
from src.agent.fdt import MinigridFDTAgent
from src.trainer import AgentTrainer
from src.dataset.seeds import LEVELS_CONFIGS
from src.constants import ENV_METADATA_PATH, OUTPUT_PATH

os.environ["WANDB_DISABLED"] = "true"
os.environ["ENV_METADATA_PATH"] = ENV_METADATA_PATH

args = get_args()

args["output"] = OUTPUT_PATH
args["run_name"] = "21-aug-test-1"
args["num_episodes"] = 20
args["seed"] = 0
args["policy"] = "random_used_action_space_only"
args["wandb_mode"] = "disabled"
args["report_to"] = "none"
args["epochs"] = 5
args["log_interval"] = 1
args["train_mode"] = "curriculum"

frame_size = 64 if args["fully_obs"] else 56

log("setting up devices")
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_str = f"{device.type}:{device.index}" if device.index else f"{device.type}"
    os.environ["CUDA_VISIBLE_DEVICES"] = device_str
    log(f"Using device: {torch.cuda.get_device_name()}")
    log("using gpu")
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print(
            "MPS not available because the current PyTorch install was not "
            "built with MPS enabled."
        )
    else:
        print(
            "MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine."
        )
        device = torch.device("cpu")
        log("using cpu")
else:
    device = torch.device("mps")
    log("using mps")

if not "single" in args["train_mode"]:
    log("Creating dataset...with multiple tasks.")
    dataset = []
    for config in list(LEVELS_CONFIGS["original_tasks"].keys())[:3]:
        args["level"] = config
        dataset.append(CustomDataset.get_dataset(args))
    args["epochs"] = max(args["epochs"], len(dataset))  # + len(datasets) // 4
else:
    log("Creating dataset...with a single task.")
    dataset = CustomDataset.get_dataset(args)

import numpy as np

features = np.zeros(64)

if "round" in args["train_mode"]:
    log("Creating round-robin collator...")
    collator = RoundRobinCollator(custom_dataset=dataset)
elif "curriculum" in args["train_mode"]:
    log(
        f"Creating {'anti' if 'anti-' in args['train_mode'] else ''}curriculum collator..."
    )
    collator = CurriculumCollator(
        custom_dataset=dataset, anti=False if "anti" in args["train_mode"] else True
    )
else:
    log("Creating standard single-task collator...")
    collator = Collator(
        custom_dataset=dataset,
        feedback=True,
        mission=True,
        context_length=args["context_length"],
    )

# for epoch in range(0, args["epochs"]):
#     batch = collator(features)
#     collator.update_epoch()

log("creating agent...")
agent = MinigridFDTAgent(
    config=DecisionTransformerConfig(
        state_dim=collator.state_dim,
        act_dim=collator.act_dim,
        state_shape=(3, frame_size, frame_size),
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
