from typing import Dict

from transformers import (
    Trainer,
    TrainingArguments,
)

from src.agent import Agent, AgentInput
from src.collator import Collator
from src.dataset import CustomDataset
from .evaluator import Evaluator


class AgentTrainer(Trainer):
    def __init__(self, args: Dict, agent: Agent, collator: Collator, dataset: CustomDataset):
        self.user_args = args

        super().__init__(
            model=agent,
            args=TrainingArguments(
                run_name=self.user_args["run_name"],
                output_dir=self.user_args["output"],
                report_to=None if self.user_args["wandb_mode"] == "disabled" else "wandb",
                logging_steps=self.user_args["log_interval"],
                remove_unused_columns=False,
                num_train_epochs=self.user_args["epochs"],
                per_device_train_batch_size=self.user_args["batch_size"],
                learning_rate=self.user_args["lr"],
                weight_decay=1e-4,
                warmup_ratio=0.1,
                optim="adamw_torch",
                max_grad_norm=0.25,
                save_strategy="no",
            ),
            train_dataset=dataset,
            data_collator=collator,
        )

        self.create_callbacks()

    def create_callbacks(self):
        self.add_callback(
            Evaluator(user_args=self.user_args, collator=self.data_collator, gamma=1)
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        input = AgentInput(**inputs)
        output = model(input)
        loss = output["loss"]
        return (loss, output) if return_outputs else loss
