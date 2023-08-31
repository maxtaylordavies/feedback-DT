import os

import torch
from transformers import DecisionTransformerConfig

from src.agent.fdt import MinigridFDTAgent
from src.collator import Collator
from src.collator import CurriculumCollator
from src.collator import RoundRobinCollator
from src.constants import ENV_METADATA_PATH
from src.constants import GLOBAL_SEED
from src.dataset.custom_dataset import CustomDataset
from src.dataset.seeds import LEVELS_CONFIGS
from src.trainer import AgentTrainer
from src.utils.argparsing import get_args
from src.utils.utils import log
from src.utils.utils import seed


os.environ["WANDB_DISABLED"] = "true"
os.environ["ENV_METADATA_PATH"] = ENV_METADATA_PATH

seed(GLOBAL_SEED)

args = get_args()

args["num_episodes"] = 10
args["wandb_mode"] = "disabled"
args["report_to"] = "none"
args["epochs"] = 5
args["log_interval"] = 1
args["policy"] = "random"


frame_size = 64 if args["fully_obs"] else 56

log("setting up devices")
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_str = f"{device.type}:{device.index}" if device.index else f"{device.type}"
    os.environ["CUDA_VISIBLE_DEVICES"] = device_str
    # log(f"Using device: {torch.cuda.get_device_name()}")
    log("using gpu")
elif not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        log(
            "MPS not available because the current PyTorch install was not "
            "built with MPS enabled."
        )
    else:
        log(
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
    for level in list(LEVELS_CONFIGS["original_tasks"].keys()):
        args["level"] = level
        dataset.append(CustomDataset.get_dataset(args))
    args["epochs"] = max(args["epochs"], len(dataset) * 2)
else:
    log("Creating dataset...with a single task.")
    dataset = CustomDataset.get_dataset(args)

if "round" in args["train_mode"]:
    log("Creating round-robin collator...")
    collator = RoundRobinCollator(custom_dataset=dataset, args=args)
elif "curriculum" in args["train_mode"]:
    log(
        f"Creating {'anti' if 'anti-' in args['train_mode'] else ''}curriculum collator..."
    )
    collator = CurriculumCollator(custom_dataset=dataset, args=args)
else:
    log("Creating standard single-task collator...")
    collator = Collator(
        custom_dataset=dataset,
        args=args,
    )

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
    dataset=collator.dataset,
    args=args,
)

log("training agent...")
trainer.train()