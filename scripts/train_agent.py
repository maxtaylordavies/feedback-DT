import os

import numpy as np
import torch
from transformers import DecisionTransformerConfig

from src.dataset.custom_dataset import CustomDataset
from src.collator import Collator
from src.agent.fdt import AtariFDTAgent
from src.trainer import AgentTrainer
from src.utils.utils import log


log("imports done")

os.environ["WANDB_DISABLED"] = "true"

DATA_DIR = "/home/s2227283/projects/feedback-DT/data/dqn_replay"
GAME = "Seaquest"
NUM_SAMPLES = 100000
CONTEXT_LENGTH = 30
BATCH_SIZE = 128
EPOCHS = 300
SEED = 123

log("setting up devices")
# device = setup_devices(SEED, useGpu=True)
log(torch.cuda.is_available())
device = torch.device("cuda")
log(f"Using device: {torch.cuda.get_device_name()}")
device_str = f"{device.type}:{device.index}" if device.index else f"{device.type}"
os.environ["CUDA_VISIBLE_DEVICES"] = device_str

log("creating dataset")
dataset = CustomDataset.from_dqn_replay(DATA_DIR, GAME, NUM_SAMPLES)

log("creating collator")
collator = Collator(custom_dataset=dataset, feedback=None, context_length=CONTEXT_LENGTH)

log("creating agent")
agent = AtariFDTAgent(
    config=DecisionTransformerConfig(
        state_dim=collator.state_dim,
        act_dim=collator.act_dim,
        state_shape=(4, 84, 84),
        max_length=CONTEXT_LENGTH,
    )
)

log("creating trainer")
trainer = AgentTrainer(
    agent=agent,
    collator=collator,
    dataset=dataset,
    args={
        "run_name": "june-19-seaquest-1",
        "env_name": "atari:Seaquest",
        "seed": SEED,
        "output": "/home/s2227283/projects/feedback-DT/data/output",
        "wandb_mode": "disabled",
        "report_to": "none",
        "log_interval": 1,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": 5e-4,
        "context_length": CONTEXT_LENGTH,
        "plot_on_train_end": True,
        "record_video": True,
    },
)

log("training")
trainer.train()
