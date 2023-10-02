#!/usr/bin/env python3
"""Script for generating experiment.txt"""
import itertools
import os
from datetime import datetime

# define some paths
USER, SCRATCH_DISK = os.environ["USER"], "/disk/scratch"
PROJECT_HOME, SCRATCH_HOME = (
    f"/home/{USER}/projects/feedback-DT",
    f"{SCRATCH_DISK}/{USER}",
)
EXPERIMENT_NAME = "conditioning_mlp_crannog"
DATA_HOME = f"{SCRATCH_HOME}/projects/feedback-DT/data/{EXPERIMENT_NAME}"


def run_name(combo, keys):
    """Create a name for the experiment based on the parameters"""
    combo_strings = "-".join(
        [
            f"{key}_{value.lower() if isinstance(value, str) else value}" if key != "model_seed" else ""
            for key, value in zip(keys, combo)
        ]
    )
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    return f"{current_datetime}-{combo_strings}".rstrip("-")


# this is the base command that will be used for the experiment
base_call = f"python {PROJECT_HOME}/scripts/train_agent_babyai.py -o {DATA_HOME}/output --load_existing_dataset True --early_stopping_patience 20"

# define a dictionary of variables to perform a grid search over.
# the key for each variable should match the name of the command-line
# argument required by the script in base_call
variables = {
    "level": [
        # "GoToObj",
        # "GoToLocal",
        # "PutNextLocal",
        # "PickupLoc",
        # "Pickup",
        # "UnblockPickup",
        # "Open",
        # "Unlock",
        # "PutNext",
        "Synth",
        # "SynthLoc"
        # "GoToSeq"
    ],
    "use_mission": [
        True,
        False
    ],
    "use_feedback": [
        True,
        # comment out False when using the "rule" and "task" feedback_mode's
        False
    ],
    "use_rtg": [
        True,
        False
    ],
    "mission_mode": [
        "standard",
        # "random"
    ],
    "feedback_mode": [
        "all",
        # comment out "rule" and "task" when using True and False for use_feedback
        # "rule",
        # "task",
        # "random"
    ],
    # "random_mode": [
    #     "english",
    #     "lorem"
    # ],
    "rgb_obs": [
        True,
        # False
    ],
    "eps_per_seed": [
        100,
    ],
     "model_seed": [
        987654321, 
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
    expt_call = f"export RUN_NAME={run_name(c, variables.keys())}; " + expt_call
    print(expt_call, file=output_file)

output_file.close()
