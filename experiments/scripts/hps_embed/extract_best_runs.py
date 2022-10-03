import optuna
import typer
import yaml


def main(
    storage_dir: str = typer.Option(...),
    output_path: str = typer.Option(...),
):
    config = {"params": {}}

    for dataset in ("WikiCS", "Amazon-CS", "Amazon-Photo", "Coauthor-CS"):
        study = optuna.load_study(
            study_name="my_study",
            storage=storage_dir + f"/{dataset}.db",
        )
        config["params"][dataset] = {
            **study.best_params,
            **{
                k: v
                for k, v in study.best_trial.user_attrs.items()
                if k != "metrics"
            },
        }

    with open(output_path, "w") as fout:
        yaml.safe_dump(data=config, stream=fout)


if __name__ == "__main__":
    typer.run(main)
