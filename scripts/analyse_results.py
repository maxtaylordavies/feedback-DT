import pandas as pd
from src.constants import OUTPUT_PATH
import os
import seaborn as sns
from matplotlib import pyplot as plt

def combine_eps_per_seed_num_train_seeds(df, multiply=False):
    if multiply:
        return df["eps_per_seed"] * df["num_train_seeds"]
    return df["eps_per_seed"].astype(str) + "_" + df["num_train_seeds"].astype(str)

def combine_conditioning(df, with_random=False, with_type=False):
    if with_type:
        feedback = (df["feebdack_mode"] + ("_" + df["random_mode"] if df["feedback_mode"] == "random" and with_random else "") + "_feedback") if df["use_feedback"] else "no_feedback"
        mission = (df["mission_mode"]+ ("_" + df["random_mode"] if df["mission_mode"] == "random" and with_random else "") + "_mission") if df["use_mission"] else "no_mission"
    rtg = "rtg" if df["use_rtg"] else "no_rtg"
    return mission + "_" + feedback + "_" + rtg

def get_experiments(output_path, dir):
    settings = dir.split("-")[5:]
    dfs = []
    for seed_dir in os.listdir(os.path.join(output_path, dir)):
        exp_path = os.path.join(output_path, os.path.join(dir, seed_dir))
        df = pd.read_pickle(os.path.join(exp_path, "results.pkl"))
        for s in settings:
            param = s.split("_")[:-1]
            param ="_".join(param)
            value = s.split("_")[-1]
            value = int(value) if value.isnumeric() else value
            df[param] = value
            try:
                df["num_eps"] = combine_eps_per_seed_num_train_seeds(df, multiply=True)
            except KeyError:
                pass
            try:
                df["eps_per_seed_num_train_seeds"] = combine_eps_per_seed_num_train_seeds(df)
            except KeyError:
                pass
            try:    
                df["conditioning"] = combine_conditioning(df)
            except KeyError:
                pass
            try: 
                df["conditioning_with_type"] = combine_conditioning(df, with_type=True)
            except KeyError:
                pass
            try:
                df["conditioning_with_random"] = combine_conditioning(df, with_type=True, with_random=True)
            except KeyError:
                pass
            try:
                df["inference"] = "mission " + df["mission_at_inference"] + "feedback " + df["feedback_at_inference"]
            except KeyError:
                pass
        dfs.append(df)
    return dfs

def get_combined_df(output_path):
    dfs = []
    for dir in os.listdir(output_path):
        if "level" in dir and not "seed" in dir:
            current_dfs = get_experiments(output_path, dir)
            dfs.extend(current_dfs)
    dfs = [df[(df["eval_type"] != "efficiency") & (df["model"] == "DT")] for df in dfs]
    comb_df = pd.concat(dfs, ignore_index=True)
    return comb_df

def plot_iid(level, level_df, param, metric, output_path):
    # iid_df = level_df[level_df["eval_type"] == "iid_generalisation"][[param, metric]].groupby(param).mean().reset_index()
    iid_df = level_df[level_df["eval_type"] == "iid_generalisation"]
    plt.figure()
    ax1 = sns.barplot(x=param, y=metric, data=iid_df)
    plt.xticks(rotation=45)
    plt.title(f"IID generalisation on {level}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path,f"iid_generalisation_{level}_{param}_{metric}.png"))
    print("SAVED FIG")
    plt.show()

def plot_ood(level, level_df, param, metric, output_path):
    # ood_df = level_df[level_df["eval_type"] == "ood_generalisation"][[param, metric, "ood_type"]].groupby([param, "ood_type"]).mean().reset_index()
    ood_df = level_df[level_df["eval_type"] == "ood_generalisation"]
    plt.figure()
    ax = sns.barplot(x=param, y=metric, hue="ood_type", data=ood_df)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.title(f"OOD generalisation on {level}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path,f"ood_generalisation_{level}_{param}_{metric}.png"))
    print("SAVED FIG")
    plt.show()

def plot_levels(output_path, params):
    comb_df = get_combined_df(output_path)
    for level in comb_df["level"].unique():
        level_df = comb_df[comb_df["level"] == level]
        for param in params:
            plot_iid(level, level_df, param, "gc_success", output_path)
            plot_ood(level, level_df, param, "gc_success", output_path)

USER = os.environ["USER"]
PROJECT_HOME = f"/home/{USER}/projects/feedback-DT"

# Experiment names should correspond to the folder name
EXPERIMENT_NAMES = [
    "random_start_or_from_end", 
    "num_seeds_vs_eps_per_seed", 
    "loss_mean", 
    "ep_distribution",
    "mission_at_inference",
    "feebdack_at_inference",
    "random_feedback",
    "random_mission",
    "conditioning"
]
# Params should correspond to existing parameters (as tested), unless they're explicitly defined in the
# get_experiments function
PARAMS = {
    "random_start_or_from_end": ["randomise_start"], 
    "num_seeds_vs_eps_per_seed": ["eps_per_seed_num_train_seeds", "num_eps"], 
    "loss_mean": ["loss_mean_type"],
    "ep_distribution": ["ep_dist"],
    "mission_at_inference": ["inference"],
    "feedback_at_inference": ["inference"],
    "random_feedback": ["conditioning_with_random"],
    "random_mission": ["conditioning_with_random"],
    "conditioning": ["conditioning_with_type"],
}
for experiment in EXPERIMENT_NAMES:
    DATA_HOME = f"{PROJECT_HOME}/data/{experiment}"
    OUTPUT = f"{DATA_HOME}/output"
    plot_levels(OUTPUT, PARAMS[experiment])



