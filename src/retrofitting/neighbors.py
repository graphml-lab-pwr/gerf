from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.utils import degree, to_undirected
from tqdm.auto import tqdm


def get_neighbors(edge_index: torch.Tensor, attr: torch.Tensor):
    unique_nodes = edge_index.flatten().unique()

    # Fit KNN
    max_neighbors = (
        (edge_index == edge_index.mode().values.unsqueeze(-1))
        .sum(dim=1)
        .max()
        .item()
    )
    knn = NearestNeighbors(n_neighbors=max_neighbors + 1)
    knn.fit(attr)

    all_attribute_neighbors = knn.kneighbors(attr, return_distance=False)

    attribute_edge_index = []

    degrees = degree(to_undirected(edge_index.clone())[0]).int()

    inv_degrees = 1. / degrees
    inv_degrees[torch.isinf(inv_degrees)] = 0

    for u in tqdm(unique_nodes, desc="Build attribute edge index", leave=False):
        attr_neighbors = torch.tensor(
            all_attribute_neighbors[u, 1:(1 + degrees[u].item())]
        )

        for an in attr_neighbors.tolist():
            attribute_edge_index.append((u.item(), an))

    attribute_edge_index = torch.tensor(attribute_edge_index).t()

    return attribute_edge_index, inv_degrees
