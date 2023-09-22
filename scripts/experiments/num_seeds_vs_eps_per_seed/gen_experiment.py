#!/usr/bin/env python3
"""Script for generating experiment.txt"""
import itertools
import os
from datetime import datetime

# define some paths
USER = os.environ["USER"]
PROJECT_HOME = f"/scratch/{USER}/projects/feedback-DT"
EXPERIMENT_NAME = "num_seeds_vs_eps_per_seed"
DATA_HOME = f"{PROJECT_HOME}/data/{EXPERIMENT_NAME}"


def run_name(combo, keys):
    """Create a name for the experiment based on the parameters"""
    combo_strings = "-".join(
        [
            f"{key}_{value.lower() if isinstance(value, str) else value}"
            for key, value in zip(keys, combo)
        ]
    )
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return f"{current_datetime}-{combo_strings}"


# this is the base command that will be used for the experiment
base_call = f"python {PROJECT_HOME}/scripts/train_agent_babyai.py -o {DATA_HOME}/output --load_existing_dataset True"

# define a dictionary of variables to perform a grid search over.
# the key for each variable should match the name of the command-line
# argument required by the script in base_call
variables = {
    "level": [
        "PutNextLocal",
        # "GoToObjMaze"
    ],
    "eps_per_seed": [
        100,
        10,
        1
    ],
    "num_train_seeds": [
        12,
        128,
        1280
    ]
}

combinations = list(itertools.product(*variables.values()))
print(f"Total experiments = {len(combinations)}")

output_file = open(
    f"{PROJECT_HOME}/scripts/experiments/{EXPERIMENT_NAME}/experiment.txt",
    "w+",
)

for c in combinations:
    expt_call = base_call
    for i, var in enumerate(variables.keys()):
        expt_call += f" --{var} {c[i]}"

    expt_call += f" --run_name {run_name(c, variables.keys())}"
    print(expt_call, file=output_file)

output_file.close()
