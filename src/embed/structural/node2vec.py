from typing import List, Optional

import numpy as np
import torch
from torch_geometric.nn import Node2Vec
from tqdm import tqdm


class Node2vecModel:
    def __init__(
        self,
        edge_index: torch.Tensor,
        emb_dim: int,
        walk_length: int,
        walks_per_node: int,
        context_size: int,
        num_negative_samples: int,
        p: float,
        q: float,
        num_nodes: int,
        batch_size: int,
        learning_rate: float,
        num_workers: int,
    ):
        self._emb_dim = emb_dim
        self._num_workers = num_workers

        self._batch_size = batch_size

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._model = Node2Vec(
            edge_index=edge_index,
            embedding_dim=emb_dim,
            walk_length=walk_length,
            walks_per_node=walks_per_node,
            context_size=context_size,
            num_negative_samples=num_negative_samples,
            p=p,
            q=q,
            num_nodes=num_nodes,
            sparse=True,
        ).to(self._device)

        self._optimizer = torch.optim.SparseAdam(
            params=list(self._model.parameters()),
            lr=learning_rate,
        )

    def fit(self, num_epochs: int) -> List[float]:
        self._model.train()

        losses = []
        pbar = tqdm(iterable=range(num_epochs), desc="Epochs")

        for _ in pbar:
            loader = self._model.loader(
                batch_size=self._batch_size,
                shuffle=True,
                num_workers=self._num_workers,
            )

            epoch_loss = 0
            for pos_rw, neg_rw in tqdm(loader, desc="Batches", leave=False):
                self._optimizer.zero_grad()

                loss = self._model.loss(
                    pos_rw=pos_rw.to(self._device),
                    neg_rw=neg_rw.to(self._device),
                )
                loss.backward()

                self._optimizer.step()

                epoch_loss += loss.item()

            losses.append(epoch_loss / len(loader))
            pbar.set_postfix(
                {
                    "first_loss": losses[0],
                    "current_loss": losses[-1],
                    "mean_loss": np.mean(losses),
                }
            )

        pbar.close()
        return losses

    def predict(self, nodes: Optional[torch.Tensor] = None) -> torch.Tensor:
        self._model.eval()
        embeddings = self._model.embedding.weight.cpu().detach()

        if nodes is None:  # Return embeddings for all nodes
            return embeddings

        # else return embeddings for given nodes
        return embeddings[nodes]
