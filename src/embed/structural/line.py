from typing import Optional

import networkx as nx
import torch
from ge import LINE


class LINEModel:
    def __init__(
        self,
        graph: nx.Graph,
        emb_dim: int,
        negative_ratio: int,
        order: str,
        batch_size: int,
    ):
        self._model = LINE(
            graph=graph,
            embedding_size=emb_dim,
            negative_ratio=negative_ratio,
            order=order,
        )
        self._batch_size = batch_size

    def fit(self, num_epochs: int) -> dict:
        hist = self._model.train(
            batch_size=self._batch_size,
            epochs=num_epochs,
            verbose=1,
        )

        losses = hist.history
        return losses

    def predict(self, nodes: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self._model.get_embeddings()

        if nodes is None:
            nodes = sorted(emb.keys())

        embeddings = torch.stack(
            [torch.tensor(emb[node], dtype=torch.float) for node in nodes],
            dim=0,
        )

        return embeddings
