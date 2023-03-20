import numpy as np

from src.argparsing import get_args
from src._datasets import get_dataset

args = get_args()

args["load_dataset_if_exists"] = False
args["seed"] = 0
args["policy"] = lambda: np.random.randint(0, 3)

for num_eps in [100, 1000, 10000, 100000]:
    args["num_episodes"] = num_eps
    dataset = get_dataset(args)
    print(len(dataset), np.max(dataset.actions))
