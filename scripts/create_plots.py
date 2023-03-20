import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np

sns.set_theme()

api = wandb.Api(timeout=30)

ids = {
    "GoToRedBallGrey": {
        100: ["lynn0rj6", "ws5ob0by", "n40n53n2", "uqo9y0ze"],
        1000: ["l26jsh7b", "stfpm2az", "bpua24xu", "3imbplw6"],
        10000: [
            "yso0ervd",
            "1z143rkb",
            "qnn5n6mp",
            "9gbijtom",
            "usrup927",
            "813j04m1",
            "cnqz6xa9",
        ],
        100000: ["7e7abva6", "9blusajw", "e0s77cml", "llm362se"],
        250000: ["qfzuhyy6", "a9izxwmr", "rlzqusyi", "34jg6dy2"],
        500000: ["58yqwwot", "5op05gp2", "cgmm8gfo", "rk6g5opy"],
    },
    "GoToRedBall": {
        100: ["wznw458e", "z3t6j3eb", "bbprze0y", "qzrga063"],
        1000: ["vbklnet7", "p8memrlb", "24lwq64u", "r29aia14"],
        10000: ["k3zbgpk5", "lm7affy4", "bjhs83y3"],
        100000: ["2atckw3p", "gsilfm1f", "776o9yem", "y8vy9m00"],
        250000: ["ygzdkhkv", "z51wa30t", "20i87uva", "la84u3bb"],
        500000: ["i8quzbhd", "22pmy7rw", "x0uxdd3h", "9rkap61i"],
    },
    "GoToRedBlueBall": {
        100: ["vz41jrjg", "zsrkqsrz", "rh6j51zv", "crwgm4gm"],
        1000: ["n2fjg3sk", "l427neav", "1ur1jeqt", "7xmz2nf8"],
        10000: ["s0sypgeo", "hectsc6e", "2iqqivsx", "xmembzb2"],
        100000: ["dz1jenjh", "yp0gjlp3", "86wj90bv", "7gcli27z"],
        250000: ["gezknq9d", "1ajjh0ig", "38wybuj5", "7zrj3dgl"],
        500000: ["s2oafwhi", "d8boutl2", "18nbe9g3", "n4k5mdu6"],
    },
    "GoToObj": {
        100: ["vl48trto", "0k3cnh5d", "wq4mcnbl", "1b4g24dn"],
        1000: ["babq7hi0", "udzcdq1e", "uwbl3ggb", "7nv8gxzy"],
        10000: ["7chop2ne", "nvx9o747", "omkss99y", "sfibfgcg"],
        100000: ["467zvfgm", "lxj2hfze", "0569b83v", "7kmdptjf"],
        250000: ["e7vosr3b", "7hw98ojh", "nl8x7f5u", "dv5bca41"],
        500000: ["s3eme7z7", "fm7mk8bt", "yrhat0m6", "zkgw6y52"],
    },
}

# "e92slfvl",


def get_returns_data(run_name):
    data = pd.read_pickle(f"../data/baseline-2/output/{run_name}/returns.pkl")
    data = data[data["epoch"] <= 10]

    data["return"] = data["return"].apply(lambda r: 100 * (r / 0.9))
    data["success"] = 100 * (data["return"] > 0)

    return pd.concat(
        [data, pd.DataFrame({"epoch": [0], "return": [0], "success": [0]})], ignore_index=True
    )


def get_data_for_n(task_name, n):
    loss_data = pd.DataFrame({"epoch": [], "loss": [], "context": []})
    return_data = pd.DataFrame({"epoch": [], "return": [], "context": []})

    for id in ids[task_name][n]:
        run = api.run(f"mlp-sam/feedback-DT/{id}")

        if "context_length" not in run.config:
            run.config["context_length"] = 1
        if "run_name" not in run.config:
            run.config["run_name"] = "friday-11"

        ld = run.history()[["train/epoch", "train/loss"]].rename(
            columns={"train/epoch": "epoch", "train/loss": "loss"}
        )
        ld = ld[ld["epoch"] <= 10]

        rd = get_returns_data(run.config["run_name"])

        ld["context"] = int(run.config["context_length"])
        rd["context"] = int(run.config["context_length"])

        loss_data = pd.concat([loss_data, ld], ignore_index=True)
        return_data = pd.concat([return_data, rd], ignore_index=True)

    return loss_data, return_data


def get_data_for_task(task_name):
    data_dict = {}
    n_vals = ids[task_name].keys()

    for n in tqdm(n_vals):
        loss_data, return_data = get_data_for_n(task_name, n)
        data_dict[n] = {"loss_data": loss_data, "return_data": return_data}

    return data_dict


def create_line_plots_for_task(task_name, data, formats=["png", "pdf"], legend=False):
    n_vals = list(data.keys())
    fig, axs = plt.subplots(2, len(n_vals), sharex=True, sharey="row", figsize=(12, 5))

    for i, n in tqdm(list(enumerate(n_vals))):
        palette = sns.color_palette(
            "flare", n_colors=len(data[n]["loss_data"]["context"].unique())
        )
        show_legend = legend and i == len(ids[task_name].keys()) - 1

        loss_ax = sns.lineplot(
            data[n]["loss_data"],
            x="epoch",
            y="loss",
            hue="context",
            errorbar=None,
            ax=axs[0, i],
            legend=show_legend,
            palette=palette,
        )
        loss_ax.set(title=f"{n} episodes", ylabel="Training loss")
        if show_legend:
            sns.move_legend(axs[0, i], "upper left", bbox_to_anchor=(1, 1))

        return_ax = sns.lineplot(
            data[n]["return_data"],
            x="epoch",
            y="return",
            hue="context",
            ax=axs[1, i],
            legend=False,
            palette=palette,
        )
        return_ax.set(ylabel="Test episode success rate (%)")

    fig.suptitle(task_name)
    fig.tight_layout()

    for format in formats:
        fig.savefig(f"{task_name}.{format}")


def create_bar_plot_for_task(task_name, data, axs, i):
    n_vals, d = list(data.keys()), []
    for n in n_vals:
        tmp = []
        for cl in [1, 4, 16, 64]:
            df = data[n]["return_data"][data[n]["return_data"]["context"] == cl]
            tmp.append(df.groupby("epoch")["return"].mean().max())
        d.append({"n": n, "return": np.max(tmp)})

    sns.barplot(data=pd.DataFrame(d), x="n", y="return", palette="flare", ax=axs[i])
    axs[i].set(
        title=task_name,
        xlabel="Size of training dataset (episodes)",
        ylabel="Best average return (%)" if i == 0 else None,
    )


def create_plots_for_tasks(task_names, formats=["svg", "pdf"]):
    fig, axs = plt.subplots(1, len(task_names), sharey=True, figsize=(12, 4))

    for i, task_name in enumerate(task_names):
        data_dict = get_data_for_task(task_name)
        create_line_plots_for_task(task_name, data_dict, legend=False, formats=formats)
        create_bar_plot_for_task(task_name, data_dict, axs, i)

    fig.tight_layout()
    for format in formats:
        fig.savefig(f"bar.{format}")


def main():
    create_plots_for_tasks(["GoToRedBallGrey", "GoToRedBall", "GoToObj"])


if __name__ == "__main__":
    main()
