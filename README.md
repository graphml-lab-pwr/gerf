# A deeper look at Graph Embedding RetroFitting
This repository provides the code used in the paper entitled *A deeper look at
Graph Embedding RetroFitting*. In this paper, we re-examine the idea of the GERF
(Graph Embedding RetroFitting) model and provide a simplification of the objective
function as well as an algorithm for hyperparameter estimation. We extend the
experimental scenario by a link prediction study. This repository contains the
whole experimental pipeline described in the paper.

![](assets/graph-embedding-retrofitting.png)


### How to use?
- create and activate virtual environment (`venv`)
- install dependencies (`pip install -r requirements.txt`)
- *pull all files from the DVC remote (`dvc pull`)

(*) The whole pipeline should be reproducible without any external data dependencies.
If you want to use precomputed stage outputs, please perform the `dvc pull` command
and it will download all stage artifacts into the `data/` directory. You don't
need any credentials as a public DVC remote endpoint is used in the DVC configuration
file. The total size of all artifacts is about 25GB.

If you want to use Docker instead of virtual environments, this repo contains also
a ready-to-use Dockerfile:
```bash
docker build -t gerf:latest -f docker/Dockerfile .

./docker/run-docker-gpu.sh "<gpu-id>"
```

## Training & evaluation
We implement all our models using the PyTorch-Geometric library and use DVC
(Data Version Control) for model versioning. DVC enables to run all experiments
in a single command and ensure better reproducibility. To reproduce the whole
pipeline run: `dvc repro` and to execute a single stage use: `dvc repro -f -s <stage name>`

There are following stages (see `dvc.yaml` file):
- `hps_embed_{node2vec,line,sdne,tadw,fscnmf,dgi}` – runs a hyperparameter search for a given node embedding method (i.e. node2vec, LINE, SDNE, ...) using the Optuna library,
- `extract_best_runs` – extracts the best found hyperparameter configurations,
- `embed_{node2vec,line,sdne}@<dataset_name>` – uses the Node2vec/LINE/SDNE method for computing structural node embeddings,
- `embed_{tadw,fscnmf,dgi}@<dataset_name>` – uses the TADW/FSCNMF/DGI method for computing attributed node embeddings,
- `apply_concat_refiner@<dataset_name>` – computes node embeddings as the naive concatenation of structural embeddings and node attributes,
- `apply_concat_pca_refiner@<dataset_name>` – similar to the above, but applies PCA to the resulting embedding vectors,
- `apply_mlp_refiner@<dataset_name>` – computes node embeddings as an MLP applied on the structural embeddings and node attributes,
- `hps_GERF@<dataset_name>` – runs a hyperparameter grid search for the improved GERF model (node classification task),
- `apply_GERF_grid@<dataset_name>` – computes node embeddings using the improved GERF model and the best hyperparamters found using the grid search,
- `apply_GERF_{uniform,homophily}@<dataset_name>` – computes node embeddings using the improved GERF model with the hyperparamters being estimated using our proposed algorithm (with uniform or homophily-based prior),
- `evaluate_node_classification@<dataset_name>` – evaluates node embeddings (specified in the configuration file) in a node classification task,
- `hps_GERF_lp@<dataset_name>` – runs a hyperparameter grid search for the improved GERF model (link prediction task),
- `apply_GERF_grid_lp@<dataset_name>` – computes node embeddings using the improved GERF model and the best hyperparamters found using the grid search (link prediction),
- `apply_GERF_{uniform,homophily}_lp@<dataset_name>` – computes node embeddings using the improved GERF model with the hyperparamters being estimated using our proposed algorithm (with uniform or homophily-based prior),
- `evaluate_link_prediction@<dataset_name>` – evaluates node embeddings (specified in the configuration file) in a link prediction task,
- `make_report_tables` - summarizes node classification and link prediction performance into a single table per task,
- `make_hps_plot@<dataset_name>` - prepares a visualization of the hyperparameter grid search (node classification).


All hyperparameters are stored in configuration files in the `experiments/configs/`
directory, whereas the experimental Python scripts are placed in the `experiments/scripts/` directory.

## License
MIT
