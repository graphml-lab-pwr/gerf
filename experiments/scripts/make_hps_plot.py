import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def make_hps_plot(hps_log_data: pd.DataFrame) -> plt.Figure:
    fig, axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15, 5),
        sharex=True,
        sharey=True,
    )

    for col, (method_key, method_name) in enumerate(
        (
            ("line", "LINE"),
            ("n2v", "Node2vec"),
            ("sdne", "SDNE"),
        )
    ):
        data = hps_log_data[hps_log_data["emb_method"] == method_key].copy()
        data["val_auc"] = data["val_auc"] * 100.

        best = data[data["val_auc"] == data["val_auc"].max()]
        best_lambda_x = best["lambda_x"].values[0]
        best_lambda_z = best["lambda_z"].values[0]
        best_auc = best["val_auc"].values[0]

        axs[col].plot(
            data["lambda_x"],
            data["val_auc"],
            marker="o",
            linestyle="--",
        )
        axs[col].set(
            title=(
                f"{method_name}\n"
                f"Best: $\lambda_Z$ = {best_lambda_z}, $\lambda_X$ = {best_lambda_x}, "
                f"AUC = {best_auc:.2f} [%]"
            ),
            xlabel="$\lambda_X$",
            ylabel="AUC [%]" if col == 0 else "",
            xticks=data["lambda_x"],
        )
        axs[col].tick_params(axis='x', rotation=45)

    fig.tight_layout()
    return fig


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", help="Name of the dataset", required=True)

    return parser.parse_args()


def main():
    args = get_args()

    sns.set("paper")

    hps_log_data = pd.read_csv(f"data/hps/GERF/{args.dataset}/log.csv")
    output_path = f"data/plots/hps_{args.dataset}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig = make_hps_plot(hps_log_data=hps_log_data)
    fig.savefig(output_path)


if __name__ == "__main__":
    main()
