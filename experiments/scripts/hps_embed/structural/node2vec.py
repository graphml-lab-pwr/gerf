"""Compute Node2vec embeddings for a given dataset."""
import os

import optuna
import typer
import yaml

from src import DATA_DIR
from src.embed.structural.node2vec import Node2vecModel
from src.hps.optimization_task import OptimizationTask
from src.tasks.node_classification import evaluate_node_classification


class Node2VecOptimizationTask(OptimizationTask):
    def _objective(self, trial: optuna.Trial) -> float:
        params = self._suggest_params(trial)

        # Build model
        n2v = Node2vecModel(
            edge_index=self.data.edge_index,
            emb_dim=params["emb_dim"],
            walk_length=params["walk_length"],
            context_size=params["context_size"],
            walks_per_node=params["walks_per_node"],
            num_negative_samples=params["num_negative_samples"],
            p=params["p"],
            q=params["q"],
            num_nodes=self.data.num_nodes,
            batch_size=params["batch_size"],
            learning_rate=params["lr"],
            num_workers=int(os.getenv("NUM_WORKERS", default=4)),
        )

        # Train model
        n2v.fit(num_epochs=params["num_epochs"])

        # Get embeddings
        z = n2v.predict()

        # Evaluate
        metrics = evaluate_node_classification(
            z=z,
            data=self.data.clone().to("cpu"),
        )
        trial.set_user_attr("metrics", metrics)
        value = metrics["val"]["auc"]
        assert isinstance(value, float)
        return value


def main(
    dataset: str = typer.Option(...), n_trials: int = typer.Option(...)
) -> None:
    # Read config
    with open(
        "experiments/configs/hps_embed/structural/node2vec.yaml", "r"
    ) as fin:
        cfg = yaml.safe_load(fin)

    params = cfg["params"][dataset]

    task = Node2VecOptimizationTask(dataset, params)
    storage_path = os.path.join(
        DATA_DIR, cfg["paths"]["output"]["storage"].replace("${name}", dataset)
    )
    task.optimize(storage_path, n_trials)


typer.run(main)
