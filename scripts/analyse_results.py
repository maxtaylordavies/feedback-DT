import os
import warnings

import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = "Helvetica"
plt.rcParams["font.family"] = "sans-serif"
sns.set(font="Helvetica")
USER = os.environ["USER"]
ROOT = os.path.dirname(os.path.abspath(USER))


def get_conditioning(row):
    """
    Returns a string describing the conditioning used for a given experiment.
    """
    conditioning = []
    if row["use_rtg"]:
        conditioning.append("rtg")
    if row["use_mission"]:
        conditioning.append("mission")
    if row["use_feedback"]:
        if row["feedback_mode"] != "all":
            conditioning.append(f"{row['feedback_mode']} feedback")
        else:
            conditioning.append("all feedback")
    return " + ".join(conditioning)


def get_experiments(output_path, dir):
    """
    Returns a list of dataframes containing the results of all experiments in a given directory.
    """
    settings = dir.split("-")[5:]
    dfs = []
    for seed_dir in os.listdir(os.path.join(output_path, dir)):
        exp_path = os.path.join(output_path, os.path.join(dir, seed_dir))
        try:
            df = pd.read_pickle(os.path.join(exp_path, "results.pkl"))
            df["model_seed"] = seed_dir
        except:
            continue
        for s in settings:
            key = s.split("_")[:-1]
            key = "_".join(key)
            value = s.split("_")[-1]
            value = (
                int(value)
                if value.isnumeric()
                else (
                    True if value == "True" else (False if value == "False" else value)
                )
            )
            df[key] = value
        try:
            df["conditioning"] = df.apply(lambda row: get_conditioning(row), axis=1)
        except KeyError:
            raise KeyError("Could not generate conditioning column")
        dfs.append(df)
    return dfs


def get_combined_df(output_path):
    """
    Returns a dataframe containing the results of all experiments in a given directory.
    """
    dfs = []
    for dir in os.listdir(output_path):
        if "level" in dir:
            current_dfs = get_experiments(output_path, dir)
            dfs.extend(current_dfs)
    dfs = [df[(df["eval_type"] != "efficiency") & (df["model"] == "DT")] for df in dfs]
    return pd.concat(dfs, ignore_index=True)


def aggr_results(df, eval_type, metric, ood_type=False):
    """
    Returns a dataframe containing the aggregated results of a given evaluation type.
    """
    if eval_type == "ood_generalisation" and ood_type:
        results_df = (
            df[df["eval_type"] == eval_type][["conditioning", "ood_type", metric]]
            .groupby(["conditioning", "ood_type"])
            .agg(
                Mean=(metric, "mean"),
                Std=(metric, "std"),
                Min=(metric, "min"),
                Max=(metric, "max"),
            )
            .sort_values("Mean", ascending=False)
        )
    else:
        results_df = (
            df[df["eval_type"] == eval_type][["conditioning", metric]]
            .groupby("conditioning")
            .agg(
                Mean=(metric, "mean"),
                Std=(metric, "std"),
                Min=(metric, "min"),
                Max=(metric, "max"),
            )
            .sort_values("Mean", ascending=False)
        )
        results_df["ood_type"] = None
    results_df["eval_type"] = eval_type
    return results_df


def get_deltas(df, reference):
    """
    Returns a dataframe containing the deltas of a given dataframe with respect to a reference (mission or RTG).
    """
    df.reset_index(inplace=True)
    delta_df = df[
        (df["conditioning"].str.contains(reference))
        | (df["conditioning"] == "all feedback")
    ]
    delta_df["Delta (Mean)"] = delta_df.apply(
        lambda row: row["Mean"]
        - delta_df.loc[delta_df["conditioning"] == reference, "Mean"].values[0],
        axis=1,
    )
    delta_df["reference"] = reference
    return delta_df


def combine_results(df, metric):
    """
    Returns a dataframe containing the aggregated results of all evaluation types.
    """
    dfs = []
    for reference in ["mission", "rtg"]:
        for eval_type in df["eval_type"].unique():
            results = aggr_results(df, eval_type, metric)
            results_with_deltas = get_deltas(results, reference)
            dfs.append(results_with_deltas)
            if eval_type == "ood_generalisation":
                results2 = aggr_results(df, eval_type, metric, ood_type=True)
                results2_with_deltas = get_deltas(results2, reference)
                dfs.append(results2_with_deltas)
    return pd.concat(dfs, ignore_index=True)


def save_results_as_csv(df, level, metric, output_path):
    """
    Saves a dataframe as a csv file.
    """
    path = os.path.join(output_path, f"results_{level}_{metric}.csv")
    df.to_csv(path)


def map_colors(df, colors):
    """
    Returns a dictionary mapping conditioning variants to colors for plots.
    """
    labels = list(df["conditioning"].unique())
    return {
        label: colors["uoe_si_colors"][i + 1]
        if "mission" in label
        else (
            colors["hwu_colors"][0]
            if label == "all feedback"
            else colors["aarg_colors"][i]
        )
        for i, label in enumerate(labels)
    }


def set_reference_performance_on_y_axis(ax, reference, reference_perf, ylims):
    """
    Sets the reference performance on the y-axis (instead of 0).
    """
    ax.set_ylim(bottom=-ylims, top=ylims)
    yticklabels = ax.get_yticks().tolist()
    new_yticklabels = [
        f"{reference_perf:.2f}"
        if float(ytick) == float(0)
        else (f"+{ytick:.2f}" if float(ytick) > float(0) else f"{ytick:.2f}")
        for ytick in yticklabels
    ]
    ax.set_yticklabels(new_yticklabels, fontsize=10)
    plt.text(
        -0.65,
        0.0075,
        reference,
        fontsize=10,
        fontweight="bold",
        ha="right",
        va="center",
    )


def set_bar_values(ax, df, ylims):
    """
    Sets the values of the bars on top of the bars (for positive deltas)
    / below the bars (for negatvie deltas).
    """
    for index, value in enumerate(df["Delta (Mean)"]):
        x_pos = index
        y_pos = value + ylims / 25 if value > 0 else value - ylims / 25
        va = "bottom" if value > 0 else "top"
        ax.text(x_pos, y_pos, f"{value:.2f}", ha="center", va=va, fontsize=10)


def set_pad_title(level, metric, eval_type):
    """
    Sets the title of the plot, with padding.
    """
    plt.title(
        f"'{level}' {metric.replace('_', ' ') + ' rate' if ('success' in metric and not 'rate' in metric) else ''} {eval_type.replace('_', ' ')}",
        fontsize=14,
        pad=14,
    )


def set_pad_axis_labels(reference):
    """
    Sets the labels of the x and y axes, with padding.
    """
    plt.xticks(fontsize=10)
    plt.xlabel("Conditioning variant", fontsize=12, labelpad=14)
    plt.ylabel(
        f"Δ goal-conditioned success rate (vs. {reference} only)",
        fontsize=12,
        labelpad=18,
    )


def set_margins():
    """
    Sets the margins of the plot.
    """
    plt.margins(x=0.05, y=0.05)


def plot_deltas(
    df,
    level,
    metric,
    colors,
    ylims,
    eval_type,
    output_path,
    reference,
    reference_perf,
    ood_type=None,
):
    """
    Plots the deltas of a given dataframe with respect to a reference (mission or RTG).
    """
    color_palette = map_colors(df, colors)
    eval_type = eval_type + " " + ood_type if ood_type else eval_type
    _, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(
        data=df,
        x="conditioning",
        y="Delta (Mean)",
        ax=ax,
        errorbar=None,
        palette=color_palette,
    )
    set_reference_performance_on_y_axis(ax, reference, reference_perf, ylims)
    set_bar_values(ax, df, ylims)
    set_pad_title(level, metric, eval_type)
    set_pad_axis_labels(reference)
    set_margins()
    plt.savefig(
        os.path.join(output_path, f"{level}_{metric}_{eval_type}_{reference}.png")
    )
    plt.show()


def plot_results(df, level, metric, colors, output_path):
    """
    Plots the results of a given dataframe.
    """
    ylims = max(abs(df["Delta (Mean)"].max()), abs(df["Delta (Mean)"].min())) * 1.2
    for reference in ["mission", "rtg"]:
        for eval_type in df["eval_type"].unique():
            for ood_type in df[df["eval_type"] == eval_type]["ood_type"].unique():
                if ood_type is None:
                    results_with_ref = df[
                        (df["eval_type"] == eval_type)
                        & (df["reference"] == reference)
                        & (pd.isnull(df["ood_type"]))
                    ]
                else:
                    results_with_ref = df[
                        (df["eval_type"] == eval_type)
                        & (df["reference"] == reference)
                        & (df["ood_type"] == ood_type)
                    ]
                reference_perf = results_with_ref[
                    results_with_ref["conditioning"] == reference
                ]["Mean"].values[0]
                results = results_with_ref[~df["conditioning"].isin(["mission", "rtg"])]
                results.sort_values("conditioning", inplace=True)
                display(results)
                plot_deltas(
                    results,
                    level,
                    metric,
                    colors,
                    ylims,
                    eval_type,
                    output_path,
                    reference,
                    reference_perf=reference_perf,
                    ood_type=ood_type,
                )


if __name__ == "__main__":
    metric = "gc_success"
    colors = {
        "aarg_colors": [
            "#2F8374",
            "#499385",
            "#63A297",
            "#7DB2A8",
            "#97C1BA",
            "#B1D1CB",
            "#CBE0DC",
            "#E5F0EE",
        ],
        "hwu_colors": [
            "#970E53",
            "#A42C69",
            "#B14A7E",
            "#BE6894",
            "#CB87A9",
            "#D8A5BF",
            "#E5C3D4",
            "#F2E1EA",
        ],
        "uoe_si_colors": [
            "#004f71",
            "#206583",
            "#407B94",
            "#6091A6",
            "#80A7B8",
            "#9FBDCA",
            "#BFD3DB",
            "#DFE9ED",
        ],
    }

    data_home = f"{ROOT}/data/conditioning/output"
    output_path = f"{ROOT}/data/conditioning/output/results"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    comb_df = get_combined_df(data_home)
    for level in comb_df["level"].unique():
        level_df = comb_df[comb_df["level"] == level]
        results_df = combine_results(level_df, metric)
        save_results_as_csv(results_df, level, metric, output_path)
        plot_results(results_df, level, metric, colors, output_path)
