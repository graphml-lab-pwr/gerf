"""Run DeepGraphInfomax (based on the example in PyTorch-Geometric)."""
import argparse
import os

import torch
from tqdm import trange
import yaml

from src import DATA_DIR
from src.datasets import load_citation_dataset
from src.embed.attributed.dgi import DGI


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-retrains", type=int, required=True)

    return parser.parse_args()


def main():
    # Read config
    with open("experiments/configs/embed/attributed/dgi.yaml", 'r') as fin:
        cfg = yaml.safe_load(fin)

    args = get_args()

    dataset_name = args.dataset
    num_retrains = args.num_retrains

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_citation_dataset(name=dataset_name).to(device)
    params = cfg["params"][dataset_name]

    for idx in trange(num_retrains, desc="Retrain"):
        # Build model
        dgi = DGI(
            num_node_features=data.num_node_features,
            emb_dim=params["emb_dim"],
            lr=params["lr"],
        )

        # Train model
        dgi.train(data=data, num_epochs=params["num_epochs"])

        # Get embeddings
        z = dgi.predict(data=data)

        # Save embeddings
        embedding_path = os.path.join(
            DATA_DIR,
            f"embeddings/attributed/dgi/{dataset_name}/emb_{idx}.pt",
        )
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        torch.save(obj=z, f=embedding_path)


if __name__ == '__main__':
    main()
