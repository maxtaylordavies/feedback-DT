import numpy as np

from src.utils.argparsing import get_args
from src.dataset.custom_dataset import CustomDataset

args = get_args()

args["load_dataset_if_exists"] = False
args["seed"] = 0
args["policy"] = "ppo"

# for num_eps in [100, 1000, 10000, 100000]:
#     args["num_episodes"] = num_eps
#     dataset = CustomDataset(args)
#     data = dataset.get_dataset()
#     print(len(data), np.max(data.actions))


dataset = CustomDataset(args)
data = dataset.get_dataset()
print(np.count_nonzero(np.asarray(data.terminations)))
