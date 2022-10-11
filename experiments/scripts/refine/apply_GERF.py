"""Apply graph retrofitting using node attributes."""
import argparse
import json
import os

import torch
from tqdm import tqdm

from src import DATA_DIR
from src.datasets import load_citation_dataset
from src.retrofitting.train import retrofit
from src.retrofitting.hyperparameters import estimate_hyperparameters


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        help="Name of the dataset",
        required=True,
    )
    parser.add_argument(
        "--hyperparameters-strategy",
        help="Which hyperparameters to use: ('grid', 'uniform')",
        choices=["grid", "uniform"],
        required=True,
    )

    return parser.parse_args()


def main():
    # Get params
    args = get_args()

    dataset = args.dataset
    hparams_strategy = args.hyperparameters_strategy

    grid_path = os.path.join(DATA_DIR, f"hps/GERF/{dataset}/best.json")

    # Load dataset
    data = load_citation_dataset(name=dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Refine
    for emb_method in tqdm(
        iterable=("n2v", "line", "sdne"),
        desc="Embedding method",
    ):
        # Read embedding
        embeddings_dir = os.path.join(
            DATA_DIR,
            f"embeddings/structural/{emb_method}/{dataset}/",
        )

        for fname in tqdm(
            iterable=sorted(os.listdir(embeddings_dir)),
            desc="Embedding",
            leave=False,
        ):
            z = torch.load(f=os.path.join(embeddings_dir, fname)).to(device)

            if hparams_strategy == "grid":
                with open(grid_path, "r") as fin:
                    lambda_x = json.load(fin)[emb_method]["lambda_x"]
            elif hparams_strategy in ("uniform",):
                lambda_x = estimate_hyperparameters(
                    data=data,
                    embedding=z,
                    prior_type=hparams_strategy,
                )["lambda_x"]
            else:
                raise ValueError(
                    f"Unknown hyperparameters strategy: '{hparams_strategy}'"
                )

            model = retrofit(
                data=data,
                embedding=z,
                lambda_x=lambda_x,
                lr=0.1,
                num_epochs=100,
            )
            z_star = model(z=z).cpu().detach()

            refined_embedding_path = os.path.join(
                DATA_DIR,
                f"embeddings/refined/GERF_{hparams_strategy}/{emb_method}/{dataset}",
                fname
            )
            os.makedirs(os.path.dirname(refined_embedding_path), exist_ok=True)

            torch.save(obj=z_star, f=refined_embedding_path)


if __name__ == "__main__":
    main()
