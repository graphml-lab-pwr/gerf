import torch

from src.retrofitting.neighbors import get_neighbors


class GRLoss:

    def __init__(
        self,
        edge_index: torch.Tensor,
        embedding: torch.Tensor,
        attributes: torch.Tensor,
        lambda_x: float,
    ):
        self.lambda_x = lambda_x
        self.lambda_z = 1 - lambda_x

        assert 0 <= lambda_x <= 1.0

        self.embedding = embedding

        res = get_neighbors(edge_index=edge_index, attr=attributes)
        device = embedding.device

        self.attribute_neighbors = res[0].to(device)
        self.inv_degrees = res[1].to(device)

    def __call__(self, z_star: torch.Tensor) -> torch.Tensor:
        # Attribute neighbors loss
        an_loss = neighborhood_loss(
            edge_index=self.attribute_neighbors,
            z_star=z_star,
            inv_degrees=self.inv_degrees,
        )

        # Invariance loss
        iv_loss = invariance_loss(z_star=z_star, z=self.embedding)

        loss = (
            self.lambda_z * iv_loss
            + self.lambda_x * an_loss
        )

        return loss


def invariance_loss(z_star: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Ensures that the new embedding `z_star` is close to the
    original embedding `z`."""
    return (z_star - z).pow(2).sum()


def neighborhood_loss(
    edge_index: torch.Tensor,
    z_star: torch.Tensor,
    inv_degrees: torch.Tensor,
) -> torch.Tensor:
    """Ensures that embedding of neighbors are close in the embedding space."""
    n_src = edge_index[0]
    n_dst = edge_index[1]

    loss = (
        (z_star[n_src] - z_star[n_dst]).pow(2).sum(dim=1)
        * inv_degrees[n_src]
    ).sum()

    return loss
