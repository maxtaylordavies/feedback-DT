#!/usr/bin/env python3
"""Script for generating experiment.txt"""

import itertools
import os

# define some paths
USER, SCRATCH_DISK = os.environ["USER"], "/disk/scratch_fast"
PROJECT_HOME, SCRATCH_HOME = (
    f"/home/{USER}/projects/feedback-DT",
    f"{SCRATCH_DISK}/{USER}",
)
DATA_HOME = f"{SCRATCH_HOME}/projects/feedback-DT/data/baseline"


def run_name(combo, keys):
    """Create a name for the experiment based on the parameters"""
    name = ""
    for i, key in enumerate(keys):
        short_key = key
        if "_" in key:
            short_key = "".join(w[0] for w in key.split("_"))
        name += f"{short_key}-{combo[i]}_"
    return name[:-1]


# this is the base command that will be used for the experiment
base_call = f"python {PROJECT_HOME}/src/train.py -o {DATA_HOME}/output --epochs 20 --wandb_mode offline --seed 42"

# define a dictionary of variables to perform a grid search over.
# the key for each variable should match the name of the command-line
# argument required by the script in base_call
variables = {
    "num_episodes": [500000],
}

combinations = list(itertools.product(*variables.values()))
print(f"Total experiments = {len(combinations)}")

output_file = open("experiment.txt", "w")

for c in combinations:
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    expt_call = base_call
    for i, var in enumerate(variables.keys()):
        expt_call += f" --{var} {c[i]}"
    expt_call += f" --run_name {run_name(c, variables.keys())}"
    print(expt_call, file=output_file)

output_file.close()
