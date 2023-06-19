import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.dataset.custom_dataset import CustomDataset
from src.collator.collator import Collator
from src.trainer import AtariEnv, AtariVisualiser
from src.utils.utils import log

DATA_DIR = "/home/s2227283/projects/feedback-DT/data/dqn_replay"
GAME = "Pong"
NUM_SAMPLES = 50000
CONTEXT_LENGTH = 30
BATCH_SIZE = 32
EPOCHS = 10
SEED = 123


def visualise_training_episode(collator, start, end, out_dir):
    states = collator.observations[start:end]
    states = states.reshape((len(states), 84, 84, 4))
    states = np.transpose(states, (0, 3, 1, 2))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(f"{out_dir}/train.mp4", fourcc, 30, (84, 84))

    for state in states:
        # state = state.reshape((4, 84, 84))
        frame = cv2.cvtColor(state[-1], cv2.COLOR_GRAY2RGB)
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        writer.write(frame)

    writer.release()

    return np.sum(collator.rewards[start:end]), end - start + 1


def visualise_eval_episode(start, end, out_dir):
    _env = AtariEnv(device, GAME.lower(), SEED)
    env = AtariVisualiser(_env, out_dir, "eval", SEED)
    _ = env.reset(seed=SEED)

    i, reward, done = start, 0, False
    while not done and i <= end:
        a = collator.actions[i]
        _, r, done = env.step(np.argmax(a))
        reward += r
        i += 1

    env.release()
    return reward, i - start + 1


log("setting up devices")
log(torch.cuda.is_available())
device = torch.device("cuda")
log(f"Using device: {torch.cuda.get_device_name()}")
device_str = f"{device.type}:{device.index}" if device.index else f"{device.type}"
os.environ["CUDA_VISIBLE_DEVICES"] = device_str

log("creating dataset")
dataset = CustomDataset.from_dqn_replay(DATA_DIR, GAME, NUM_SAMPLES)

log("creating collator")
collator = Collator(custom_dataset=dataset, feedback=None, context_length=CONTEXT_LENGTH)

num_eps = len(collator.episode_starts)
for i in tqdm(range(num_eps)):
    out_dir = f"./debugging/{GAME}/{i}"
    os.makedirs(out_dir, exist_ok=True)
    start, end = collator.episode_starts[i], collator.episode_ends[i] + 1
    expected_reward, expected_length = visualise_training_episode(
        collator, start, end, out_dir
    )
    reward, length = visualise_eval_episode(start, end, out_dir)
    log(
        f"(episode {i}): reward={reward} (expected {expected_reward}), length={length} (expected {expected_length})",
        with_tqdm=True,
    )
