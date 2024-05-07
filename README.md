# Dropout in GNNs

This is the code repository for a study on behavior of dropout techniques in GNNs, being worked on towards partial fulfilment of the requirements for the degree of Master of Science in Machine Learning from University College London.

## Directory Structure

- `assets` - plots generated from different experiments.
- `config` - configuration files for different datasets and models.
- `data` - raw datasets store.
- `data_classes` - Classes for handling different datasets, and make them suitable for training, eg. Cora, Citeseer, PubMed, etc.
- `dropout_classes` - Classes defining layers corresponding to different dropout methods, eg. Dropout, DropNode, DropEdge, etc.
- `model_classes` - Classes defining blocks of GNN layers, eg. GCN, GAT, etc.
- `results` - results of the different runs. <br>
    - `directory structure` - `<dataset>` -> `<model>` -> `<run-date>` -> `logs` and `<data-split>_results`
- `utils` - utility functions for running the transformer experiments.
- `main.py` - main file for training the models.
- `inference.py` - main file for testing the models.

## Setup

```bash
conda create --name <env-name> --file requirements.txt python=3.8
conda activate <env-name>
```

## Execution

To run the transformer experiments, execute
```bash
python3 -B main.py \
    --dataset <dataset> \
    --model <model>
```
You can also override default configurations using the command line.<br>

For inference, execute
```bash
python3 -B inference.py \
    --dataset <dataset> \
    --model <model> \
    --weights <path-to-weights>
```

Note: Make sure to set the device index to <i>None</i> if you do not wish to use the GPU.