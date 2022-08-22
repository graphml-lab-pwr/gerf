"""Find graph retrofitting parameters using grid search."""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src import DATA_DIR
from src.datasets import load_citation_dataset
from src.retrofitting.train import retrofit
from src.tasks.node_classification import evaluate_node_classification


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        help="Name of the dataset",
        required=True,
    )

    return parser.parse_args()


def evaluate_retrofitting(
    emb_method: str,
    dataset: str,
    lambda_x: float,
    lr: float,
    num_epochs: int,
) -> float:
    """Load all embeddings of a given graph and compute mean val AUC."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_citation_dataset(name=dataset)

    embedding_dir = os.path.join(
        DATA_DIR,
        f"embeddings/structural/{emb_method}/{dataset}/",
    )

    aucs = []

    for fname in sorted(os.listdir(embedding_dir))[:1]:
        z = torch.load(f=os.path.join(embedding_dir, fname)).to(device)

        model = retrofit(
            data=data,
            embedding=z,
            lambda_x=lambda_x,
            lr=lr,
            num_epochs=num_epochs,
        )
        z_star = model(z=z).cpu().detach()

        metrics = evaluate_node_classification(z=z_star, data=data)
        val_auc = metrics["val"]["auc"]

        aucs.append(val_auc)

    return np.mean(aucs).item()


def main():
    # Get params
    args = get_args()

    dataset = args.dataset

    # Parameter grid
    lr = 0.1
    num_epochs = 100
    step = 0.05
    lambda_xs = np.arange(0, 1 + step, step)

    output_dir = os.path.join(DATA_DIR, "hps/GERF/", dataset)
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "log.csv")
    best_path = os.path.join(output_dir, "best.json")

    results = []
    best_results = {}

    for emb_method in tqdm(
        iterable=("line", "n2v", "sdne"),
        desc="Embedding method",
    ):
        best_val_auc = -1
        best_lambda_x = None

        for lambda_x in tqdm(lambda_xs, desc="Lambda_X", leave=False):
            mean_val_auc = evaluate_retrofitting(
                emb_method=emb_method,
                dataset=dataset,
                lambda_x=lambda_x,
                lr=lr,
                num_epochs=num_epochs,
            )

            results.append({
                "emb_method": emb_method,
                "lambda_x": lambda_x,
                "lambda_z": 1 - lambda_x,
                "val_auc": mean_val_auc,
            })

            # Save hyperparameter search results
            (
                pd.DataFrame
                .from_records(results)
                .to_csv(path_or_buf=log_path, index=False)
            )

            if mean_val_auc > best_val_auc:
                best_val_auc = mean_val_auc
                best_lambda_x = lambda_x

        best_results[emb_method] = {
            "val_auc": best_val_auc,
            "lambda_x": best_lambda_x,
            "lambda_z": 1 - best_lambda_x,
        }

        with open(best_path, "w") as fout:
            json.dump(obj=best_results, fp=fout, indent=4)


if __name__ == "__main__":
    main()
