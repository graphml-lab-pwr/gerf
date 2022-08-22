import traceback
from abc import abstractmethod
from pathlib import Path
from typing import Dict, Any

import optuna
import torch
from torch_geometric.data import Data

from src.datasets import load_citation_dataset


class OptimizationTask:
    def __init__(self, dataset_name: str, params: Dict[str, Dict[str, Any]]):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = params
        self.data: Data = load_citation_dataset(name=dataset_name).to(device)

    def _suggest_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        result = dict()
        for param_name in self.params.keys():
            param_cfg = self.params[param_name]
            if param_cfg["param_type"] == "searchable":
                if param_cfg["type"] == "categorical":
                    result[param_name] = trial.suggest_categorical(
                        name=param_name, choices=param_cfg["choices"]
                    )
                elif param_cfg["type"] == "float_log":
                    result[param_name] = trial.suggest_float(
                        name=param_name,
                        low=param_cfg["low"],
                        high=param_cfg["high"],
                        log=True,
                    )
                elif param_cfg["type"] == "int":
                    result[param_name] = trial.suggest_int(
                        name=param_name,
                        low=param_cfg["low"],
                        high=param_cfg["high"],
                    )
                elif param_cfg["type"] == "float":
                    result[param_name] = trial.suggest_float(
                        name=param_name,
                        low=param_cfg["low"],
                        high=param_cfg["high"],
                    )
            elif param_cfg["param_type"] == "constant":
                result[param_name] = param_cfg["value"]
                trial.set_user_attr(param_name, result[param_name])
        return result

    def optimize(self, storage_path: str, n_trials: int) -> None:
        Path(storage_path).parent.mkdir(parents=True, exist_ok=True)

        study = optuna.create_study(
            direction="maximize",
            study_name="my_study",
            storage="sqlite:///" + str(storage_path),
        )

        study.optimize(self.objective, n_trials=n_trials, catch=(Exception,))

    def objective(self, trial: optuna.Trial) -> float:
        try:
            return self._objective(trial)
        except Exception as e:
            trial.set_system_attr("traceback", traceback.format_exc())
            raise e

    @abstractmethod
    def _objective(self, trial: optuna.Trial) -> float:
        pass
