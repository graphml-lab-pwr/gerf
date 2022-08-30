"""Run DeepGraphInfomax (based on the example in PyTorch-Geometric)."""
from pathlib import Path
from typing import Any, Dict

import optuna
import torch
import typer
import yaml
from torch_geometric.data import Data

from src import DATA_DIR
from src.datasets import load_citation_dataset
from src.embed.attributed.dgi import DGI
from src.tasks.node_classification import evaluate_node_classification


class OptimizationTask:

    def __init__(self, dataset_name: str, params: Dict[str, Dict[str, Any]]):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data: Data = load_citation_dataset(name=dataset_name).to(device)
        self.params = params

    def _suggest_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        result = dict()
        for param_name in self.params.keys():
            param_cfg = self.params[param_name]
            if param_cfg["param_type"] == "searchable":
                if param_cfg["type"] == "categorical":
                    result[param_name] = trial.suggest_categorical(
                        name=param_name,
                        choices=param_cfg["choices"])
                elif param_cfg["type"] == "log_uniform":
                    result[param_name] = trial.suggest_loguniform(
                        name=param_name,
                        low=param_cfg["low"],
                        high=param_cfg["high"])
            elif param_cfg["param_type"] == "constant":
                result[param_name] = param_cfg["value"]
        return result

    def objective(self, trial):
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
        return metrics['val']['macro avg']['f1-score']


def main(dataset: str = typer.Option(...),
         n_trials: int = typer.Option(...)):
    # Read config
    with open("experiments/configs/hps_embed/attributed/dgi.yaml", 'r') as fin:
        cfg = yaml.safe_load(fin)

    params = cfg["params"][dataset]

    task = OptimizationTask(dataset, params)
    storage_path = Path(
        DATA_DIR,
        cfg["paths"]["output"]["storage"]
        .replace("${name}", dataset)
    )
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(direction="maximize",
                                study_name="my_study",
                                storage="sqlite:///" + str(storage_path))

    study.optimize(task.objective, n_trials=n_trials)


typer.run(main)
