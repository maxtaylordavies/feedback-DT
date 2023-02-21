import os
from datetime import datetime

from datasets import load_dataset
import gym
from transformers import (
    DecisionTransformerConfig,
    Trainer,
    TrainingArguments,
)
import numpy as np
import torch

from src.argparsing import get_training_args
from src.data import DecisionTransformerGymDataCollator
from src.dt import TrainableDT
from src.utils import log, setup_devices
from src.gymrecorder import Recorder


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
    # initialise the trainer
    training_args = TrainingArguments(
        run_name=args["run_name"],
        output_dir=args["output"],
        remove_unused_columns=False,
        num_train_epochs=args["epochs"],
        per_device_train_batch_size=args["batch_size"],
        learning_rate=args["lr"],
        weight_decay=1e-4,
        warmup_ratio=0.1,
        optim="adamw_torch",
        max_grad_norm=0.25,
    )
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


def visualise_trained_model(args, collator, model):
    # create the output directory if it doesn't exist
    output_dir = os.path.join(args["output"], args["run_name"])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # build an environment to visualise the trained model
    env = gym.make("HalfCheetah-v4", render_mode="rgb_array")
    env = Recorder(env, output_dir, filename="model.mp4", fps=30)
    max_ep_len, scale = 1000, 1000.0  # normalization for rewards/returns
    target_return = (
        12000 / scale
    )  # evaluation is conditioned on a return of 12000, scaled accordingly
    device = "cpu"
    model = model.to(device)

    state_mean = collator.state_mean.astype(np.float32)
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = collator.state_std.astype(np.float32)
    state_std = torch.from_numpy(state_std).to(device=device)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    episode_return, episode_length, state = 0, 0, env.reset()

    target_return = torch.tensor(
        target_return, device=device, dtype=torch.float32
    ).reshape(1, 1)
    states = (
        torch.from_numpy(state[0])
        .reshape(1, state_dim)
        .to(device=device, dtype=torch.float32)
    )
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    for t in range(max_ep_len):
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states - state_mean) / state_std,
            actions,
            rewards,
            target_return,
            timesteps,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        pred_return = target_return[0, -1] - (reward / scale)
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)],
            dim=1,
        )

        episode_return += reward
        episode_length += 1

        if done:
            break

    # log("Saving video...")
    # env.save()


def main(args):
    # do some setup
    if not args["run_name"]:
        args["run_name"] = f"dt-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        log(f"run_name not specified, using {args['run_name']}")
    setup_devices(not args["no_gpu"], args["seed"])

    # load the data and create the model
    dataset, collator, model = load_data_and_create_model()

    # train the model
    model = train_model(args, dataset, collator, model)

    # visualise the trained model
    visualise_trained_model(args, collator, model)


if __name__ == "__main__":
    args = get_training_args()
    log(f"parsed args: {args}")
    main(args)
