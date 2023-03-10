from itertools import accumulate
import os

import numpy as np
from tqdm import tqdm

from src.argparsing import get_args
from src.utils import setup_devices
from src._datasets import get_dataset
from src.train import create_collator_and_model, train_model
from src.visualiser import visualise_trained_model, visualise_episode

os.environ["WANDB_PROJECT"] = "feedback-DT"
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["WANDB_WATCH"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_MODE"] = "dryrun"

args = get_args()

args["run_name"] = "modified-collator-test"
args["wandb_mode"] = "offline"
args["output"] = "/home/s2227283/projects/feedback-DT/data/baseline/output"
args["num_episodes"] = 10000
args["seed"] = 42
args["epochs"] = 20

print(args)

# setup compute devices
setup_devices(args["seed"], not args["no_gpu"])

# create or load training dataset
dataset = get_dataset(args)

ep_indices = []
for i in tqdm(range(len(dataset))):
    if dataset[i].rewards[-1] > 0:
        ep_indices.append(i)
ep_indices = np.array(ep_indices[:100])


def rtg(episode, gamma):
    return np.array(list(accumulate(episode.rewards[::-1], lambda a, b: (gamma * a) + b)))[
        ::-1
    ]


discounted_returns = np.array([rtg(dataset[i], 1) for i in ep_indices])
for i in range(100):
    print(ep_indices[i], discounted_returns[i][0])


# output_dir = os.path.join(args["output"], "rewarding_episodes")
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# for idx in tqdm(ep_indices):
#     ep_return = visualise_episode(dataset[idx], idx, args, output_dir)

# # create the data collator and model
# collator, model = create_collator_and_model(dataset)

# # train the model
# model = train_model(args, dataset, collator, model)

# # visualise the trained model
# returns, repeats = [0, 1, 100, 1000, 10000], 5
# for rtg in returns:
#     for rep in range(repeats):
#         visualise_trained_model(args, collator, model, epochs_trained=f"{rtg}-{rep}", target_return=rtg)
