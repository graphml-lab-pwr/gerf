import random

import numpy as np
import pandas as pd
import torch
from sklearn import linear_model as sk_lm
from sklearn import metrics as sk_mtr
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling


def evaluate_link_prediction(z: torch.Tensor, data: Data) -> dict:
    metrics = {}
    lp_model = sk_lm.LogisticRegression(n_jobs=-1, max_iter=250)

    edge_data, labels = get_edges(z, data)

    for split in ("train", "val", "test"):
        y_true = labels[split]
        X = edge_data[split]

        if split == "train":
            lp_model.fit(X=X, y=y_true)

        y_score = lp_model.predict_proba(X=X).transpose()[1]
        y_pred = lp_model.predict(X=X)

        metrics[split] = calc_metrics(
            y_score=y_score,
            y_pred=y_pred,
            y_true=y_true,
        )

    return metrics


def get_edges(z, data):
    random.seed(42)
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        method='sparse'
    )
    pos_edge_index = data.edge_index

    pos_embs = z[pos_edge_index[0]] * z[pos_edge_index[1]]
    pos_labels = torch.ones(pos_edge_index.shape[1])

    neg_embs = z[neg_edge_index[0]] * z[neg_edge_index[1]]
    neg_labels = torch.zeros(neg_edge_index.shape[1])

    embs = torch.cat([pos_embs, neg_embs], dim=0)
    labels = torch.cat([pos_labels, neg_labels], dim=0)

    return split_edges(embs, labels)


def split_edges(embs: torch.Tensor, labels: torch.Tensor):
    X_rest, X_test, y_rest, y_test = train_test_split(
        embs,
        labels,
        test_size=0.8,
        random_state=42,
        stratify=labels,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest,
        y_rest,
        test_size=0.5,
        random_state=42,
        stratify=y_rest,
    )

    edge_data = {'train': X_train, 'val': X_val, 'test': X_test}
    labels = {'train': y_train, 'val': y_val, 'test': y_test}
    return edge_data, labels


def calc_metrics(y_score, y_pred, y_true):
    """Composes multiple metrics from Scikit-learn into single dictionary.

    The computed metrics are: precision, recall, f1 per class, the AUC score
    and the confusion matrix.
    """
    metrics = {
        **sk_mtr.classification_report(
            y_true=y_true,
            y_pred=y_pred,
            output_dict=True,
        ),
        "auc": sk_mtr.roc_auc_score(
            y_true=y_true,
            y_score=y_score,
            average="macro",
        ),
        "cm": sk_mtr.confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
        ).tolist(),
    }

    return metrics
