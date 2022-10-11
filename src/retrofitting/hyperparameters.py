from typing import Tuple, Dict

import numpy as np
import torch
import torch_geometric.data
from sklearn.linear_model import LogisticRegression
from torch_geometric.data import Data


def estimate_hyperparameters(
    data: Data,
    embedding: torch.Tensor,
    prior_type: str,
) -> Dict[str, float]:
    """Estimates hyperparameters with DirichletMultinomialModel."""
    attr_dataset = _get_attr_dataset(data)
    attr_correct_preds = classify_and_count_correct_preds(*attr_dataset)

    embedding_dataset = _get_embedding_dataset(data, embedding)
    embedding_correct_preds = classify_and_count_correct_preds(*embedding_dataset)

    prior_coefficients = get_prior_values(prior_type=prior_type)
    likelihood_values = (embedding_correct_preds, attr_correct_preds)

    return dict(zip(
        ["lambda_z", "lambda_x"],
        calc_Dirichlet_Multinomial_MAP(
            prior_coefficients=prior_coefficients,
            likelihood_values=likelihood_values,
        ),
    ))


def get_prior_values(prior_type: str) -> Tuple[float, float]:
    if prior_type == "uniform":
        return 1.0, 1.0
    elif prior_type == "homophily":
        raise NotImplementedError()
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
) -> ...:
    """Calculates the MAP estimate of the Dirichlet-Multinomial model.

    Based on: https://gitlab.com/fildne/fildne/-/blob/master/dgem/embedding/incremental/estimators.py
    """
    alpha = np.array(prior_coefficients)
    N = np.array(likelihood_values)
    dim = 2

    _map = (N + alpha - 1) / (sum(N) + sum(alpha) - dim)
    return _map


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
