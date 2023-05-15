import os
import sys

from src.dataset.minari_storage import list_local_datasets, name_dataset, delete_dataset
from src.utils.argparsing import get_args


def delete_datasets(args):
    if args["del_all"]:
        local_datasets = list_local_datasets()
        if local_datasets != []:
            for d in local_datasets:
                delete_dataset(d)
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
        delete_dataset(dataset_name)


if __name__ == "__main__":
    args = get_args()
    delete_datasets(args)
