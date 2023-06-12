import os

from transformers import DecisionTransformerConfig

from src.dataset.custom_dataset import MinariDataset
from src.collator import Collator
from src.agent.fdt import AtariFDTAgent
from src.trainer import AgentTrainer
from src.utils.utils import setup_devices, log

log("imports done")

os.environ["WANDB_DISABLED"] = "true"

DATA_DIR = "/home/s2227283/projects/feedback-DT/data/dqn_replay"
GAME = "Breakout"
NUM_SAMPLES = 10000
CONTEXT_LENGTH = 30
BATCH_SIZE = 32
EPOCHS = 10
SEED = 123

log("setting up devices")
device = setup_devices(SEED, useGpu=True)

log("creating dataset")
dataset = MinariDataset.from_dqn_replay(DATA_DIR, GAME, NUM_SAMPLES)

log("creating collator")
collator = Collator(
    custom_dataset=dataset, feedback=False, context_length=CONTEXT_LENGTH
)

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
        "run_name": "breakout-test-1",
        "env_name": "atari:Breakout",
        "seed": SEED,
        "output": "/home/s2227283/projects/feedback-DT/data/output",
        "wandb_mode": "disabled",
        "report_to": "none",
        "log_interval": 10,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": 1e-4,
        "context_length": CONTEXT_LENGTH,
        "plot_on_train_end": True,
        "record_video": True,
    },
)

log("training")
trainer.train()
