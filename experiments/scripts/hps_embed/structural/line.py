"""LINE embedding using the GE (Graph Embedding) package."""
import os

import optuna
import typer
import yaml
from torch_geometric.utils import to_networkx

from src import DATA_DIR
from src.embed.structural.line import LINEModel
from src.hps.optimization_task import OptimizationTask
from src.tasks.node_classification import evaluate_node_classification


class LINEOptimizationTask(OptimizationTask):
    def _objective(self, trial: optuna.Trial) -> float:
        params = self._suggest_params(trial)

        graph = to_networkx(data=self.data, to_undirected=True)

        # Build model
        line = LINEModel(
            graph=graph,
            emb_dim=params["emb_dim"],
            negative_ratio=params["negative_ratio"],
            order=params["order"],
            batch_size=params["batch_size"],
        )

        # Train model
        line.fit(num_epochs=params["num_epochs"])

        # Get embeddings
        z = line.predict()

        if z.isnan().any():
            raise ValueError

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
        "experiments/configs/hps_embed/structural/line.yaml", "r"
    ) as fin:
        cfg = yaml.safe_load(fin)

    params = cfg["params"][dataset]

    task = LINEOptimizationTask(dataset, params)
    storage_path = os.path.join(
        DATA_DIR, cfg["paths"]["output"]["storage"].replace("${name}", dataset)
    )
    task.optimize(storage_path, n_trials)


typer.run(main)
