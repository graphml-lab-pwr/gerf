"""Run FSCNMF method to obtain attributed node embeddings."""
import argparse
import os

import torch
from torch_geometric.utils import to_networkx
from tqdm.auto import trange
import yaml

from src import DATA_DIR
from src.datasets import load_citation_dataset
from src.embed.attributed.fscnmf import _FSCNMF


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-retrains", type=int, required=True)

    return parser.parse_args()


def main():
    # Read config
    with open("experiments/configs/embed/attributed/fscnmf.yaml", 'r') as fin:
        cfg = yaml.safe_load(fin)

    args = get_args()

    dataset_name = args.dataset
    num_retrains = args.num_retrains

    data = load_citation_dataset(name=dataset_name)
    params = cfg["params"][dataset_name]

    for idx in trange(num_retrains, desc="Retrain"):
        graph = to_networkx(data=data, to_undirected=True)

        # Build model
        fscnmf = _FSCNMF(
            dimensions=params["emb_dim"],
            iterations=params["num_epochs"],
            alpha_1=params["alpha_1"],
            alpha_2=params["alpha_2"],
            alpha_3=params["alpha_3"],
            beta_1=params["beta_1"],
            beta_2=params["beta_2"],
            beta_3=params["beta_3"],
        )

        fscnmf.allow_disjoint = True

        # Train model
        fscnmf.fit(graph=graph, X=data.x.numpy())

        # Get embeddings
        z = torch.from_numpy(fscnmf.get_embedding()).float()

        # Save embeddings
        embedding_path = os.path.join(
            DATA_DIR,
            f"embeddings/attributed/fscnmf/{dataset_name}/emb_{idx}.pt",
        )
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        torch.save(obj=z, f=embedding_path)


if __name__ == '__main__':
    main()
