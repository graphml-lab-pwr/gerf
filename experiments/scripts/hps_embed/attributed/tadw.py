"""Run TADW method to obtain attributed node embeddings."""
import os

import optuna
import torch
import typer
import yaml
from torch_geometric.utils import to_networkx

from src import DATA_DIR
from src.embed.attributed.tadw import _TADW
from src.hps.optimization_task import OptimizationTask
from src.tasks.node_classification import evaluate_node_classification


class TADWOptimizationTask(OptimizationTask):
    def _objective(self, trial: optuna.Trial) -> float:
        params = self._suggest_params(trial)

        graph = to_networkx(data=self.data, to_undirected=True)

        # Build model
        tadw = _TADW(
            dimensions=params["emb_dim"],
            iterations=params["num_epochs"],
            alpha=params["lr"],
            reduction_dimensions=params["reduction_dimensions"],
            svd_iterations=params["svd_iterations"],
            lambd=params["lambd"],
        )

        tadw.allow_disjoint = True

        # Train model
        tadw.fit(graph=graph, X=self.data.x.numpy())

        # Get embeddings
        z = torch.from_numpy(tadw.get_embedding()).float()

        # Evaluate
        metrics = evaluate_node_classification(z=z, data=self.data)
        trial.set_user_attr("metrics", metrics)
        value = metrics["val"]["auc"]
        assert isinstance(value, float)
        return value


def main(
    dataset: str = typer.Option(...), n_trials: int = typer.Option(...)
) -> None:
    # Read config
    with open(
        "experiments/configs/hps_embed/attributed/tadw.yaml", "r"
    ) as fin:
        cfg = yaml.safe_load(fin)

    params = cfg["params"][dataset]

    task = TADWOptimizationTask(dataset, params)
    storage_path = os.path.join(
        DATA_DIR, cfg["paths"]["output"]["storage"].replace("${name}", dataset)
    )
    task.optimize(storage_path, n_trials)


typer.run(main)
