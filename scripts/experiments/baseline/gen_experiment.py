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
DATA_HOME = f"{SCRATCH_HOME}/projects/feedback-DT/data/baseline-3"


def run_name(combo, keys):
    """Create a name for the experiment based on the parameters"""
    return f"dummy-feedback-redballgrey-{combo[0]}-{combo[1]}-{combo[2]}"
    # name = "friday_" + combo[0].split("-")[1] + "_"
    # for i, key in enumerate(list(keys)[1:]):
    #     short_key = key
    #     if "_" in key:
    #         short_key = "".join(w[0] for w in key.split("_"))
    #     name += f"{short_key}-{combo[i + 1]}_"
    # return name[:-1]


# this is the base command that will be used for the experiment
base_call = f"python {PROJECT_HOME}/src/train.py -o {DATA_HOME}/output --epochs 10 --wandb_mode offline"

# define a dictionary of variables to perform a grid search over.
# the key for each variable should match the name of the command-line
# argument required by the script in base_call
variables = {
    "num_episodes": [10000, 100000],
    "feedback_type": ["direction", "distance"],
    "feedback_freq_steps": [1],
}

combinations = list(itertools.product(*variables.values()))
print(f"Total experiments = {len(combinations)}")

output_file = open("experiment.txt", "w")

for c in combinations:
    expt_call = base_call
    for i, var in enumerate(variables.keys()):
        expt_call += f" --{var} {c[i]}"

    expt_call += f" --randomise_starts {False} --log_interval {10} --seed {42} --run_name {run_name(c, variables.keys())}"
    print(expt_call, file=output_file)

output_file.close()
