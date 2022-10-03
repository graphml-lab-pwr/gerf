"""LINE embedding using the GE (Graph Embedding) package."""
import argparse
import os

import torch
import yaml
from torch_geometric.utils import to_networkx
from tqdm import trange

from src import DATA_DIR
from src.datasets import load_citation_dataset
from src.embed.structural.line import LINEModel


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-retrains", type=int, required=True)

    return parser.parse_args()


def main():
    # Read config
    with open("experiments/configs/embed/structural/line.yaml", 'r') as fin:
        cfg = yaml.safe_load(fin)

    args = get_args()

    dataset_name = args.dataset
    num_retrains = args.num_retrains

    data = load_citation_dataset(name=dataset_name)
    params = cfg["params"][dataset_name]

    for idx in trange(num_retrains, desc="Retrain"):
        graph = to_networkx(data=data, to_undirected=True)

        # Build model
        line = LINEModel(
            graph=graph,
            emb_dim=params["emb_dim"],
            negative_ratio=params["negative_ratio"],
            order=params["order"],
            batch_size=params["batch_size"],
        )

        # Train model
        line.fit(num_epochs=params["num_epochs"])

        # Get embeddings
        z = line.predict()

        if z.isnan().any():
            exit(1)

        # Save embeddings
        embedding_path = os.path.join(
            DATA_DIR,
            f"embeddings/structural/line/{dataset_name}/emb_{idx}.pt",
        )
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        torch.save(obj=z, f=embedding_path)


if __name__ == '__main__':
    main()
