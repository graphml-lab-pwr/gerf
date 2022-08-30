from typing import Union

import networkx as nx
import numpy as np
from karateclub import FSCNMF
from scipy.sparse import coo_matrix
from tqdm.auto import trange


class _FSCNMF(FSCNMF):

    def fit(
        self,
        graph: nx.classes.graph.Graph,
        X: Union[np.array, coo_matrix],
    ):
        """This implementation adds a progress bar."""
        self._set_seed()
        self._check_graph(graph)

        self._X = X
        self._A = self._create_base_matrix(graph)

        self._init_weights()

        for _ in trange(self.iterations, desc="Iterations"):
            self._update_B1()
            self._update_B2()
            self._update_U()
            self._update_V()

    def _create_D_inverse(self, graph):
        """
        Creating a sparse inverse degree matrix.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.

        Return types:
            * **D_inverse** *(Scipy array)* - Diagonal inverse degree matrix.
        """
        index = np.arange(graph.number_of_nodes())
        values = np.array([
            1.0/graph.degree[node] if graph.degree[node] > 0 else 0.0
            for node in range(graph.number_of_nodes())
        ])
        shape = (graph.number_of_nodes(), graph.number_of_nodes())
        D_inverse = coo_matrix((values, (index, index)), shape=shape)
        return D_inverse
