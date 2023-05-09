import json

import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np

from src.utils import log

sns.set_theme()

api = wandb.Api(timeout=30)

history_cols = {
    "train/epoch": "epoch",
    "train/loss": "loss",
}

config_cols = {
    "context": "context_length",
    "feedback type": "feedback_type",
    "feedback freq": "feedback_freq_steps",
}

save_dir = "../figures"


def get_runs():
    all_runs = api.runs(
        path="mlp-sam/feedback-DT",
    )
    print(len(all_runs))

    def _get_runs(task):
        task = task.replace("GoTo", "").lower()

        filtered = []
        for run in all_runs:
            if "feedback" not in run.name:
                continue
            if run.name.split("-")[1].replace("goto", "") != task:
                continue

            run = _fill_in_config(run)
            filtered.append(run)

        print(len(filtered))
        return filtered

    def _fill_in_config(run):
        if "run_name" in run.config:
            return run

        run.config["run_name"] = run.name
        _, _, num_eps, feedback_type, feedback_freq, = run.config[
            "run_name"
        ].split("-")

        if "num_episodes" not in run.config:
            run.config["num_episodes"] = int(num_eps)
        if "feedback_type" not in run.config:
            run.config["feedback_type"] = feedback_type
        if "feedback_freq_steps" not in run.config:
            run.config["feedback_freq_steps"] = int(feedback_freq)

        run.update()
        return run

    tasks = ["GoToRedBallGrey", "GoToRedBall", "GoToObj"]
    ids = {task: {} for task in tasks}

    for task in tasks:
        print(f"Getting runs for {task}...")
        for run in tqdm(_get_runs(task)):
            num_eps = run.config["num_episodes"]
            if num_eps not in ids[task]:
                ids[task][num_eps] = []
            ids[task][num_eps].append(run.id)

    with open("./feedback_ids.json", "w") as f:
        json.dump(ids, f)


def format_num(num):
    return num if num < 1 else int(num)


def get_returns_data_for_run(run):
    try:
        data = pd.read_pickle(f"../data/output/{run.config['run_name']}/returns.pkl")
        data = data[data["epoch"] <= 10]

        data["return"] = data["return"].apply(lambda r: 100 * (r / 0.9))
        data["success"] = 100 * (data["return"] > 0)

        return pd.concat(
            [pd.DataFrame({"epoch": [0], "return": [0], "success": [0]}), data],
            ignore_index=True,
        )
    except:
        return pd.DataFrame({"epoch": [0], "return": [0], "success": [0]})


def get_loss_data_for_run(run):
    ld = run.history()[list(history_cols.keys())].rename(columns=history_cols)
    return ld[ld["epoch"] <= 10]


def get_data_for_n(ids, task_name, n):
    loss_data = pd.DataFrame(
        {x: [] for x in list(history_cols.values()) + list(config_cols.keys())}
    )
    return_data = pd.DataFrame(
        {x: [] for x in ["epoch", "return", "success"] + list(config_cols.keys())}
    )

    for id in ids[task_name][n]:
        run = api.run(f"mlp-sam/feedback-DT/{id}")

        ld = get_loss_data_for_run(run)
        rd = get_returns_data_for_run(run)

        for k, v in config_cols.items():
            ld[k] = run.config.get(v, None)
            rd[k] = run.config.get(v, None)

        loss_data = pd.concat([loss_data, ld], ignore_index=True)
        return_data = pd.concat([return_data, rd], ignore_index=True)

    return loss_data, return_data


def get_data_for_task(ids, task_name):
    data_dict, n_vals = {}, ids[task_name].keys()

    for n in tqdm(n_vals):
        loss_data, return_data = get_data_for_n(ids, task_name, n)
        data_dict[n] = {"loss_data": loss_data, "return_data": return_data}

    return data_dict


def create_line_plots_for_task(
    ids,
    task_name,
    data,
    x="epoch",
    loss_y="loss",
    return_y="return",
    hue="context",
    hue_order=None,
    style=None,
    style_order=None,
    formats=["svg", "pdf"],
    legend=False,
    prefix="",
    palette_name="flare",
    **kwargs,
):
    n_vals = sorted(list(data.keys()))
    fig, axs = plt.subplots(
        3 if style else 2,
        len(n_vals),
        sharex=True,
        sharey="row",
        figsize=(12, 7.5 if style else 5),
    )

    for i, n in enumerate(n_vals):
        palette = sns.color_palette(
            palette_name, n_colors=len(data[n]["loss_data"][hue].unique())
        )
        show_legend = legend and i == len(ids[task_name].keys()) - 1

        loss_ax = sns.lineplot(
            data[n]["loss_data"],
            x=x,
            y=loss_y,
            hue=hue,
            hue_order=hue_order or data[n]["loss_data"][hue].unique(),
            style=style,
            errorbar=None,
            ax=axs[0, i],
            legend=show_legend,
            palette=palette,
        )
        loss_ax.set(title=f"{format_num(int(n)/1000)}k episodes", ylabel="Training loss")

        if show_legend:
            sns.move_legend(axs[0, i], "upper left", bbox_to_anchor=(1, 1))

        return_ax = sns.lineplot(
            data[n]["return_data"],
            x=x,
            y=return_y,
            hue=hue,
            hue_order=hue_order or data[n]["return_data"][hue].unique(),
            errorbar="se",
            ax=axs[1, i],
            legend=False,
            palette=palette,
        )
        return_ax.set(ylabel="Test return (%)", ylim=(0, 100))

        if style:
            return_ax_2 = sns.lineplot(
                data[n]["return_data"],
                x=x,
                y=return_y,
                style=style,
                style_order=style_order or data[n]["return_data"][style].unique(),
                errorbar="se",
                ax=axs[2, i],
                legend=False,
            )
            return_ax_2.set(ylabel="Test return (%)", ylim=(0, 100))

    fig.suptitle(task_name)
    fig.tight_layout()

    for format in formats:
        fig.savefig(f"{save_dir}/{prefix}_{task_name}.{format}")


def create_bar_plot_for_task(
    task_name, baseline_data, feedback_data, axs, i, return_y="return", **kwargs
):
    n_vals, d = sorted(list(feedback_data.keys())), []

    for n in n_vals:
        bd, fd = baseline_data[n]["return_data"], feedback_data[n]["return_data"]

        for type_val in ["action", "adjacency", "distance", "direction"]:
            tmp = []
            for freq_val in fd["feedback freq"].unique():
                _max = (
                    fd[(fd["feedback type"] == type_val) & (fd["feedback freq"] == freq_val)]
                    .groupby("epoch")[return_y]
                    .mean()
                    .max()
                )
                if not np.isnan(_max):
                    tmp.append(_max)
            if tmp:
                d.append({"n": n, return_y: max(tmp), "model": type_val})

        # also record average of all feedback types
        feedback_maxes = [x[return_y] for x in d if x["n"] == n and x["model"] != "baseline"]
        d.append({"n": n, return_y: np.mean(feedback_maxes), "model": "feedback avg"})

        bd_best = bd[bd["context"] == 64].groupby("epoch")[return_y].mean().max()
        d.append({"n": n, return_y: bd_best, "model": "baseline"})

    sns.barplot(
        data=pd.DataFrame(d),
        x="n",
        y=return_y,
        hue="model",
        ax=axs[i],
        palette=sns.color_palette("husl", 6),
        saturation=0.8,
        linewidth=0,
    )

    axs[i].set(
        title=task_name,
        xticklabels=[f"{format_num(int(n) / 1000)}k" for n in n_vals],
        xlabel="Size of training dataset (episodes)",
        ylabel="Best average return (%)" if i == 0 else None,
        ylim=(0, 100),
    )
    axs[i].legend_.remove()


def main():
    get_runs()

    with open("./baseline_ids.json", "r") as f:
        baseline_ids = json.load(f)
    with open("./feedback_ids.json", "r") as f:
        feedback_ids = json.load(f)

    ids = {"baseline": baseline_ids, "feedback": feedback_ids}
    tasks = ["GoToRedBallGrey", "GoToRedBall", "GoToObj"]

    fig, axs = plt.subplots(1, len(tasks), sharey=True, figsize=(16, 4))

    for i, task in enumerate(tasks):
        print(f"Processing task {task}...")
        data = {
            "baseline": get_data_for_task(baseline_ids, task),
            "feedback": get_data_for_task(feedback_ids, task),
        }
        create_line_plots_for_task(
            ids["baseline"],
            task,
            data["baseline"],
            legend=False,
            prefix="baseline",
            hue="context",
            style=None,
        )
        create_line_plots_for_task(
            ids["feedback"],
            task,
            data["feedback"],
            legend=False,
            prefix="feedback",
            hue="feedback freq",
            style="feedback type",
            palette_name="mako",
        )
        create_bar_plot_for_task(task, data["baseline"], data["feedback"], axs, i)

    fig.tight_layout()
    fig.savefig(f"{save_dir}/bars.svg")
    fig.savefig(f"{save_dir}/bars.pdf")


if __name__ == "__main__":
    main()
