"""SDNE embedding using the GE (Graph Embedding) package."""
import os

import optuna
import typer
import yaml
from torch_geometric.utils import to_networkx

from src import DATA_DIR
from src.embed.structural.sdne import SDNEModel
from src.hps.optimization_task import OptimizationTask
from src.tasks.node_classification import evaluate_node_classification


class SDNEOptimizationTask(OptimizationTask):
    def _objective(self, trial: optuna.Trial) -> float:
        params = self._suggest_params(trial)

        graph = to_networkx(data=self.data, to_undirected=True)

        # Build model
        line = SDNEModel(
            graph=graph,
            emb_dim=params["emb_dim"],
            hidden_size=params["hidden_size"],
            alpha=params["alpha"],
            beta=params["beta"],
            nu1=params["nu1"],
            nu2=params["nu2"],
            batch_size=params["batch_size"],
        )

        # Train model
        line.fit(num_epochs=params["num_epochs"])

        # Get embeddings
        z = line.predict()

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
        "experiments/configs/hps_embed/structural/sdne.yaml", "r"
    ) as fin:
        cfg = yaml.safe_load(fin)

    params = cfg["params"][dataset]

    task = SDNEOptimizationTask(dataset, params)
    storage_path = os.path.join(
        DATA_DIR, cfg["paths"]["output"]["storage"].replace("${name}", dataset)
    )
    task.optimize(storage_path, n_trials)


typer.run(main)
