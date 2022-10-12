from typing import Tuple, Dict, Optional

import networkx as nx
import numpy as np
import torch
import torch_geometric.data
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from src.retrofitting.neighbors import get_neighbors


def estimate_hyperparameters(
    data: Data,
    embedding: torch.Tensor,
    prior_type: str,
    **prior_kwargs,
) -> Dict[str, float]:
    """Estimates hyperparameters with DirichletMultinomialModel."""
    attr_dataset = _get_attr_dataset(data)
    attr_correct_preds = classify_and_count_correct_preds(*attr_dataset)

    embedding_dataset = _get_embedding_dataset(data, embedding)
    embedding_correct_preds = classify_and_count_correct_preds(*embedding_dataset)

    prior_coefficients = get_prior_values(
        prior_type=prior_type, data=data, embeddings=embedding, **prior_kwargs
    )
    likelihood_values = (embedding_correct_preds, attr_correct_preds)

    return dict(zip(
        ["lambda_z", "lambda_x"],
        calc_Dirichlet_Multinomial_MAP(
            prior_coefficients=prior_coefficients,
            likelihood_values=likelihood_values,
        ),
    ))


def get_prior_values(
    prior_type: str,
    data: Optional[Data] = None,
    embeddings: Optional[torch.Tensor] = None,
    prior_sample_frac: float = 0.05,
) -> Tuple[float, float]:
    if prior_type == "uniform":
        return 1.0, 1.0
    elif prior_type == "homophily":
        assert data is not None and embeddings is not None
        prior_sample_size = len(data.x) * prior_sample_frac
        alpha_z, alpha_x = compute_attr_and_structural_homophily(data, embeddings)
        return int(alpha_z * prior_sample_size), int(alpha_x * prior_sample_size)
    else:
        raise ValueError(f"Unknown prior type: '{prior_type}'")


def classify_and_count_correct_preds(x_train, y_train, x_val, y_val):
    """Trains classifier and returns #correct predictions on test set."""
    clf = LogisticRegression(n_jobs=-1, max_iter=250)
    clf.fit(x_train, y_train)

    preds = clf.predict(x_val)
    return np.sum(preds == y_val)


def calc_Dirichlet_Multinomial_MAP(
    prior_coefficients: Tuple[float, float],
    likelihood_values: Tuple[int, int],
) -> np.ndarray:
    """Calculates the MAP estimate of the Dirichlet-Multinomial model.

    Based on: https://gitlab.com/fildne/fildne/-/blob/master/dgem/embedding/incremental/estimators.py
    """
    alpha = np.array(prior_coefficients)
    N = np.array(likelihood_values)
    dim = 2

    _map = (N + alpha - 1) / (sum(N) + sum(alpha) - dim)
    return _map


def compute_attr_and_structural_homophily(
    data: Data, embeddings: torch.Tensor
) -> Tuple[float, float]:
    homophily_z = compute_homophily(data.edge_index, embeddings)
    homophily_x = compute_homophily(data.edge_index, data.x)
    return homophily_z, homophily_x


def compute_homophily(edge_index: torch.Tensor, features: torch.Tensor) -> float:
    """Homophily defined as a fraction of nearest neighbors wrt attributes being network neighbors."""

    attribute_edge_index, _ = get_neighbors(edge_index, features)

    attribute_edges = set(list(zip(*attribute_edge_index.tolist())))
    network_edges = set(list(zip(*edge_index.tolist())))
    homophily = len(attribute_edges.intersection(network_edges)) / len(network_edges)

    return homophily


def _cosine_sim_to_angular_sim(cosine_similarity: np.ndarray) -> np.ndarray:
    """Cosine similarity to angular similarity, bounded between 0 and 1.
    Source: https://en.wikipedia.org/wiki/Cosine_similarity#Angular_distance_and_similarity
    """
    return 1 - np.arccos(cosine_similarity) / np.pi


def _get_attr_dataset(
    data: torch_geometric.data.Data,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert hasattr(data, "train_mask")
    assert hasattr(data, "val_mask")

    return (
        data.x[data.train_mask].cpu().numpy(),
        data.y[data.train_mask].cpu().numpy(),
        data.x[data.val_mask].cpu().numpy(),
        data.y[data.val_mask].cpu().numpy(),
    )


def _get_embedding_dataset(
    data: torch_geometric.data.Data, embedding: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert hasattr(data, "train_mask")
    assert hasattr(data, "val_mask")

    return (
        embedding[data.train_mask].cpu().numpy(),
        data.y[data.train_mask].cpu().numpy(),
        embedding[data.val_mask].cpu().numpy(),
        data.y[data.val_mask].cpu().numpy(),
    )
