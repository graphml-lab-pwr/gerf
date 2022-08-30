"""Run TADW method to obtain attributed node embeddings."""
import argparse
import os

import torch
from torch_geometric.utils import to_networkx
from tqdm import trange
import yaml

from src import DATA_DIR
from src.datasets import load_citation_dataset
from src.embed.attributed.tadw import _TADW


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-retrains", type=int, required=True)

    return parser.parse_args()


def main():
    # Read config
    with open("experiments/configs/embed/attributed/tadw.yaml", 'r') as fin:
        cfg = yaml.safe_load(fin)

    args = get_args()

    dataset_name = args.dataset
    num_retrains = args.num_retrains

    data = load_citation_dataset(name=dataset_name)
    params = cfg["params"][dataset_name]

    for idx in trange(num_retrains, desc="Retrain"):
        graph = to_networkx(data=data, to_undirected=True)

        # Build model
        tadw = _TADW(
            dimensions=params["emb_dim"],
            iterations=params["num_epochs"],
            alpha=params["lr"],
        )

        tadw.allow_disjoint = True

        # Train model
        tadw.fit(graph=graph, X=data.x.numpy())

        # Get embeddings
        z = torch.from_numpy(tadw.get_embedding()).float()

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
