"""Run FSCNMF method to obtain attributed node embeddings."""
import os

import optuna
import torch
import typer
import yaml
from torch_geometric.utils import to_networkx

from src import DATA_DIR
from src.embed.attributed.fscnmf import _FSCNMF
from src.hps.optimization_task import OptimizationTask
from src.tasks.node_classification import evaluate_node_classification


class FSCBNMFOptimizationTask(OptimizationTask):
    def _objective(self, trial: optuna.Trial) -> float:
        params = self._suggest_params(trial)

        graph = to_networkx(data=self.data, to_undirected=True)

        # Build model
        fscnmf = _FSCNMF(
            dimensions=params["emb_dim"],
            iterations=params["num_epochs"],
            alpha_1=params["alpha_1"],
            alpha_2=params["alpha_2"],
            alpha_3=params["alpha_3"],
            beta_1=params["beta_1"],
            beta_2=params["beta_2"],
            beta_3=params["beta_3"],
        )

        fscnmf.allow_disjoint = True

        # Train model
        fscnmf.fit(graph=graph, X=self.data.x.numpy())

        # Get embeddings
        z = torch.from_numpy(fscnmf.get_embedding()).float()

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
        "experiments/configs/hps_embed/attributed/fscnmf.yaml", "r"
    ) as fin:
        cfg = yaml.safe_load(fin)
    params = cfg["params"][dataset]
    task = FSCBNMFOptimizationTask(dataset, params)
    storage_path = os.path.join(
        DATA_DIR, cfg["paths"]["output"]["storage"].replace("${name}", dataset)
    )
    task.optimize(storage_path, n_trials)


typer.run(main)
