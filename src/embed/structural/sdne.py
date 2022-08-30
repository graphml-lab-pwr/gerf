from typing import Optional

import networkx as nx
import torch
from ge import SDNE


class SDNEModel:

    def __init__(
        self,
        graph: nx.Graph,
        emb_dim: int,
        hidden_size: int,
        alpha: float,
        beta: float,
        nu1: float,
        nu2: float,
        batch_size: int,
    ):
        self._model = SDNE(
            graph=graph,
            hidden_size=[hidden_size, emb_dim],
            alpha=alpha,
            beta=beta,
            nu1=nu1,
            nu2=nu2,
        )
        self._batch_size = batch_size

    def fit(self, num_epochs: int) -> dict:
        hist = self._model.train(
            batch_size=self._batch_size,
            epochs=num_epochs,
            verbose=2,
        )

        losses = hist.history
        return losses

    def predict(
        self,
        nodes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        emb = self._model.get_embeddings()

        if nodes is None:
            nodes = sorted(emb.keys())

        embeddings = torch.stack([
            torch.tensor(emb[node], dtype=torch.float)
            for node in nodes
        ], dim=0)

        return embeddings
