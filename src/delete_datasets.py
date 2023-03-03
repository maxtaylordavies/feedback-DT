import minari
from argparsing import delete_dataset_args
from get_datasets import name_dataset, list_local_datasets

args = delete_dataset_args()
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
    num_episodes = args["num_episodes"]
    include_timeout = args["include_timeout"]
    dataset_name = name_dataset(env_name, num_episodes, include_timeout)
    minari.delete_dataset(dataset_name)
    print(f"Sucess! Deleted {dataset_name}")
