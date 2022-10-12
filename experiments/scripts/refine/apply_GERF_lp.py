"""Apply graph retrofitting using node attributes (link prediction)."""
import argparse
import json
import os
import random
from typing import Tuple

import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
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


def _convert_to_lp_problem(
    data: Data,
    z: torch.Tensor,
    seed: int = 42,
) -> Tuple[Data, torch.Tensor]:
    random.seed(seed)
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        method='sparse'
    )
    pos_edge_index = data.edge_index

    pos_labels = torch.ones(pos_edge_index.shape[1])
    neg_labels = torch.zeros(neg_edge_index.shape[1])

    labels = torch.cat([pos_labels, neg_labels], dim=0)

    rest_idx, test_idx = train_test_split(
        torch.arange(len(labels)),
        test_size=0.8,
        random_state=seed,
        stratify=labels,
    )
    train_idx, val_idx = train_test_split(
        rest_idx,
        test_size=0.5,
        random_state=seed,
        stratify=labels[rest_idx],
    )

    data_lp = Data(
        x=torch.cat([
            data.x[pos_edge_index[0]] * data.x[pos_edge_index[1]],
            data.x[neg_edge_index[0]] * data.x[neg_edge_index[1]],
        ], dim=0),
        y=torch.cat([pos_labels, neg_labels], dim=0),
        train_mask=train_idx,
        val_mask=val_idx,
    )
    z_lp = torch.cat([
        z[pos_edge_index[0]] * z[pos_edge_index[1]],
        z[neg_edge_index[0]] * z[neg_edge_index[1]],
    ], dim=0)

    return data_lp, z_lp


def main():
    # Get params
    args = get_args()

    dataset = args.dataset
    hparams_strategy = args.hyperparameters_strategy

    grid_path = os.path.join(DATA_DIR, f"hps/GERF_lp/{dataset}/best.json")

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
                data_lp, z_lp = _convert_to_lp_problem(data, z)
                lambda_x = estimate_hyperparameters(
                    data=data_lp,
                    embedding=z_lp,
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
                f"embeddings/refined/GERF_{hparams_strategy}_lp/{emb_method}/{dataset}",
                fname
            )
            os.makedirs(os.path.dirname(refined_embedding_path), exist_ok=True)

            torch.save(obj=z_star, f=refined_embedding_path)


if __name__ == "__main__":
    main()
