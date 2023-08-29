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
DATA_HOME = f"{SCRATCH_HOME}/projects/feedback-DT/data/feedback-1"


def run_name(combo, keys):
    """Create a name for the experiment based on the parameters"""
    level = combo[0].split("-")[1].lower()
    return f"feedback-{level}-{combo[1]}-{combo[2]}-{combo[3]}"


# this is the base command that will be used for the experiment
base_call = f"python {PROJECT_HOME}/scripts/train_agent_babyai.py"

# define a dictionary of variables to perform a grid search over.
# the key for each variable should match the name of the command-line
# argument required by the script in base_call
variables = {
    "level": [
        "GoToLocal",
        "PutNextLocal",
        "GoTo",
        "PutNext",
        "GoToSeq",
        "BossLevel",
    ],
    "num_episodes": [100000, 250000, 500000],
    "batch_size": [32, 64, 128],
    "context_length": [8, 16, 32, 64],
}

combinations = list(itertools.product(*variables.values()))
print(f"Total experiments = {len(combinations)}")

output_file = open("experiment.txt", "w")

for c in combinations:
    expt_call = base_call
    for i, var in enumerate(variables.keys()):
        expt_call += f" --{var} {c[i]}"

    expt_call += f" --run_name {run_name(c, variables.keys())}"
    print(expt_call, file=output_file)

output_file.close()
