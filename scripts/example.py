import os

from src.agent import RandomAgent
from src.dataset import CustomDataset
from src.collator import Collator
from src.trainer import AgentTrainer

os.environ["WANDB_DISABLED"] = "true"

NUM_EPS = 100
EP_LENGTH = 20
STATE_DIM = 128
ACT_DIM = 4
CONTEXT_LENGTH = 1

agent = RandomAgent(act_dim=ACT_DIM)
dataset = CustomDataset.random(
    num_eps=NUM_EPS, ep_length=EP_LENGTH, state_dim=STATE_DIM, act_dim=ACT_DIM
)
collator = Collator(custom_dataset=dataset, feedback=None, context_length=1)
trainer = AgentTrainer(
    agent=agent,
    collator=collator,
    dataset=dataset,
    args={
        "run_name": "dummy",
        "env_name": "ALE/Breakout-ram-v5",
        "seed": 42,
        "output": ".",
        "wandb_mode": "disabled",
        "log_interval": 10,
        "epochs": 10,
        "batch_size": 10,
        "lr": 1e-4,
        "context_length": CONTEXT_LENGTH,
        "plot_on_train_end": True,
        "record_video": False,
    },
)
trainer.train()
