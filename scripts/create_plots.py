import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

sns.set_theme()

api = wandb.Api(timeout=20)
ids = {
    "GoToRedBallGrey": {
        100: ["4oytpfuu", "ksm76b8x", "qdvau11k", "g2d8hcoh"],
        1000: ["0c2wlty6", "2eljb7fq", "h78r6t7q", "986uxc5a"],
        10000: ["2k558yi3", "d0g2chz1", "67fo752v", "5a2026rm"],
        100000: ["a8fpepa3", "8sqxdgtj", "mqdbitnp", "wtcxbnsj"],
    },
    "GoToRedBall": {
        100: ["30szurpp", "9wk2jvqw", "qowa625r", "22cy18jr"],
        1000: ["019txg8n", "6fto48qb", "ht69iv2n", "56vt9mkt"],
        10000: ["oadd3ry4", "0xg8bcq6", "83wzon8g", "pi5wjeu7"],
        100000: ["qwej6bum", "75003qid", "z96y03u2", "fl095nb5"],
    },
    "GoToRedBlueBall": {
        100: ["ac7ddyn6", "bjfkh24d", "ynynx034", "2klc4i5y"],
        1000: ["iqwnyicl", "26ix6qqa", "i3y01i98", "uo4mtsoc"],
        10000: ["mzip0wew", "e5o1ymbt", "easnvgwf", "slhqmv34"],
        100000: ["hr6j38fb", "mthzyss1", "p41ew92v", "4h83t7gg"],
    },
}


def get_returns_data(run_name, target_return=1000):
    data = pd.read_pickle(f"../data/baseline/output/{run_name}/returns.pkl")
    return data[data["target_return"] == target_return][["epoch", "return"]]

def get_data_for_n(task_name, n):
    loss_data = pd.DataFrame({"epoch": [], "loss": [], "context_length": []})
    return_data = pd.DataFrame({"epoch": [], "return": [], "context_length": []})

    for id in ids[task_name][n]:
        run = api.run(f"mlp-sam/feedback-DT/{id}")

        ld = run.history()[["train/epoch", "train/loss"]].rename(
            columns={"train/epoch": "epoch", "train/loss": "loss"}
        )
        rd = get_returns_data(run.config["run_name"])

        ld["context_length"] = run.config["context_length"]
        rd["context_length"] = run.config["context_length"]

        loss_data = pd.concat([loss_data, ld], ignore_index=True)
        return_data = pd.concat([return_data, rd], ignore_index=True)

    return loss_data, return_data


def plot_data_for_n(task_name, n, i, axs):
    loss_data, return_data = get_data_for_n(task_name, n)
    palette = sns.color_palette("flare", len(ids[task_name][n]))
    show_legend = i == len(ids[task_name].keys()) - 1

    loss_ax = sns.lineplot(
        loss_data,
        x="epoch",
        y="loss",
        hue="context_length",
        ax=axs[0, i],
        legend=show_legend,
        palette=palette,
    )
    loss_ax.set(title=f"{n} episodes", ylabel="Training loss")
    if show_legend:
        sns.move_legend(axs[0, i], "upper left", bbox_to_anchor=(1, 1))

    return_ax = sns.lineplot(
        return_data,
        x="epoch",
        y="return",
        hue="context_length",
        ax=axs[1, i],
        legend=False,
        palette=palette,
    )
    return_ax.set(ylabel="Test return")


def create_plot_for_task(task_name, formats=["png", "pdf"]):
    n_vals = ids[task_name].keys()

    fig, axs = plt.subplots(2, len(n_vals), sharex=True, sharey="row", figsize=(12, 6))
    for i, n in tqdm(list(enumerate(n_vals))):
        plot_data_for_n(task_name, n, i, axs)

    fig.suptitle(task_name)
    plt.tight_layout()

    for format in formats:
        plt.savefig(f"{task_name}.{format}")
    plt.show()


def main():
    for task in ["GoToRedBall"]:
        create_plot_for_task(task)


if __name__ == "__main__":
    main()
