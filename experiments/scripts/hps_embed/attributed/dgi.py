"""Run DeepGraphInfomax (based on the example in PyTorch-Geometric)."""
import os.path

import optuna
import typer
import yaml

from src import DATA_DIR
from src.embed.attributed.dgi import DGI
from src.hps.optimization_task import OptimizationTask
from src.tasks.node_classification import evaluate_node_classification


class DGIOptimizationTask(OptimizationTask):
    def _objective(self, trial: optuna.Trial) -> float:
        params = self._suggest_params(trial)

        # Build model
        dgi = DGI(
            num_node_features=self.data.num_node_features,
            emb_dim=params["emb_dim"],
            lr=params["lr"],
        )

        # Train model
        dgi.train(data=self.data, num_epochs=5)

        # Get embeddings
        z = dgi.predict(data=self.data)

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
    with open("experiments/configs/hps_embed/attributed/dgi.yaml", "r") as fin:
        cfg = yaml.safe_load(fin)

    params = cfg["params"][dataset]

    task = DGIOptimizationTask(dataset, params)
    storage_path = os.path.join(
        DATA_DIR, cfg["paths"]["output"]["storage"].replace("${name}", dataset)
    )
    task.optimize(storage_path, n_trials)


typer.run(main)
