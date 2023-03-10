import os
import sys

import minari

from _datasets import list_local_datasets, name_dataset
from argparsing import get_args

basepath = os.path.dirname(os.path.dirname(os.path.abspath("")))
if not basepath in sys.path:
    sys.path.append(basepath)


def delete_datasets(args):
    if args["del_all"]:
        local_datasets = list_local_datasets()
        if local_datasets != []:
            for d in local_datasets:
                minari.delete_dataset(d)
                print(f"Sucess! Deleted {d}")
            print(f"Sucess! Deleted all {len(local_datasets)} local datasets")
        else:
            print("No datasets to delete")
    else:
        env_name = args["env_name"]
        assert (
            env_name
        ), "Specify the corresponding environment to delete a specific dataset or pass True for del_all to delete all local datasets"

        num_episodes = args["num_episodes"]
        assert (
            num_episodes
        ), "Specify the corresponding number of episodes to delete a specific dataset or pass True for del_all to delete all local datasets"

        include_timeout = args["include_timeout"]
        assert (
            include_timeout
        ), "Specify the corresponding include_timeout parameter value to delete a specific dataset or pass True for del_all to delete all local datasets"

        dataset_name = name_dataset(args)
        minari.delete_dataset(dataset_name)
        print(f"Sucess! Deleted {dataset_name}")


if __name__ == "__main__":
    args = get_args()
    delete_datasets(args)
