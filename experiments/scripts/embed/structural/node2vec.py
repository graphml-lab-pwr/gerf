"""Compute Node2vec embeddings for a given dataset."""
import argparse
import os

import torch
import yaml
from tqdm import trange

from src import DATA_DIR
from src.datasets import load_citation_dataset
from src.embed.structural.node2vec import Node2vecModel


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-retrains", type=int, required=True)

    return parser.parse_args()


def main():
    # Read config
    with open("experiments/configs/embed/structural/node2vec.yaml", 'r') as fin:
        cfg = yaml.safe_load(fin)

    args = get_args()

    dataset_name = args.dataset
    num_retrains = args.num_retrains

    data = load_citation_dataset(name=dataset_name)
    params = cfg["params"][dataset_name]

    for idx in trange(num_retrains, desc="Retrain"):
        # Build model
        n2v = Node2vecModel(
            edge_index=data.edge_index,
            emb_dim=params["emb_dim"],
            walk_length=params["walk_length"],
            context_size=params["context_size"],
            walks_per_node=params["walks_per_node"],
            num_negative_samples=params["num_negative_samples"],
            p=params["p"],
            q=params["q"],
            num_nodes=data.num_nodes,
            batch_size=params["batch_size"],
            learning_rate=params["lr"],
            num_workers=int(os.getenv("NUM_WORKERS", default=4)),
        )

        # Train model
        n2v.fit(num_epochs=params["num_epochs"])

        # Get embeddings
        z = n2v.predict()

        # Save embeddings
        embedding_path = os.path.join(
            DATA_DIR,
            cfg["paths"]["output"]["embedding"]
            .replace("${name}", dataset_name)
            .replace("${idx}", str(idx)),
        )
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        torch.save(obj=z, f=embedding_path)


if __name__ == '__main__':
    main()
