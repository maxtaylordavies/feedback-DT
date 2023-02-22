#!/usr/bin/env python3
"""Script for generating experiment.txt"""

import itertools
import os

# define some paths
USER, SCRATCH_DISK = os.environ["USER"], "/disk/scratch"
PROJECT_HOME, SCRATCH_HOME = (
    f"/home/{USER}/projects/feedback-DT",
    f"{SCRATCH_DISK}/{USER}",
)
DATA_HOME = f"{SCRATCH_HOME}/projects/feedback-DT/data/baseline"

# this is the base command that will be used for the experiment
base_call = f"python {PROJECT_HOME}/src/train.py -o {DATA_HOME}/output"

# define a dictionary of variables to perform a grid search over.
# the key for each variable should match the name of the command-line
# argument required by the script in base_call
variables = {
    "epochs": [100],
}

combinations = list(itertools.product(*variables.values()))
print(f"Total experiments = {len(combinations)}")

output_file = open("experiment.txt", "w")

for c in combinations:
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    expt_call = base_call
    for i, var in enumerate(variables.keys()):
        expt_call += f" --{var}={c[i]}"
    print(expt_call, file=output_file)

output_file.close()