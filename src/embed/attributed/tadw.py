from typing import Union

import networkx as nx
import numpy as np
from karateclub import TADW
from scipy.sparse import coo_matrix
from tqdm import trange


class _TADW(TADW):

    def fit(
        self,
        graph: nx.classes.graph.Graph,
        X: Union[np.array, coo_matrix],
    ):
        """This implementation adds a progress bar and more verbosity."""
        self._set_seed()
        self._check_graph(graph)

        print("Creating target matrix A...")
        self._A = self._create_target_matrix(graph)

        print("Creating reduced features T...")
        self._T = self._create_reduced_features(X)

        self._init_weights()

        for _ in trange(self.iterations, desc="Iterations"):
            self._update_W()
            self._update_H()
