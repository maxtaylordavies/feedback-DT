import os
import subprocess
import shutil

from tqdm import tqdm

# get list of folders in current directory starting with "offline-run"
wandb_dir = "/home/s2227283/projects/feedback-DT/wandb"
runs = [
    f
    for f in os.listdir(wandb_dir)
    if os.path.isdir(os.path.join(wandb_dir, f)) and f.startswith("offline-run")
]

print(f"Found {len(runs)} runs to sync")

# for each run, call wandb sync
for run in tqdm(runs):
    result = subprocess.run(
        ["wandb", "sync", os.path.join(wandb_dir, run)], capture_output=True
    )

    output = result.stdout.decode("utf-8")
    if "done" in output:
        shutil.rmtree(os.path.join(wandb_dir, run))
    else:
        print("Error syncing run", output)