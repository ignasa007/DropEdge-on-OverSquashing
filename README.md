# Dropout in GNNs

This is the code repository for a study on behavior of dropout techniques in GNNs, being worked on towards partial fulfilment of the requirements for the degree of Master of Science in Machine Learning from University College London.

## Directory Structure

- `assets` &minus; plots generated from different experiments.
- `config` &minus; configuration files for different datasets and models.
- `data` &minus; raw datasets store.
- `data_classes` &minus; classes for handling different datasets, and make them suitable for training, eg. Cora, Citeseer, PubMed, etc.
- `dropout_classes` &minus; classes defining layers corresponding to different dropout methods, eg. Dropout, DropNode, DropEdge, etc.
- `model_classes` &minus; classes defining blocks of GNN layers, eg. GCN, GAT, etc.
- `results` &minus; results of the different runs. <br>
    - `directory structure` &minus; `<dataset>` &#8594; `<model>` &#8594; `<run-date>` &#8594; `logs` and `<data-split>_results`
- `utils` &minus; utility functions for running the transformer experiments.
- `main.py` &minus; main file for training the models.
- `inference.py` &minus; main file for testing the models.

## Setup

```bash
conda create --name <env-name> --file requirements.txt python=3.8
conda activate <env-name>
```

## Execution

To run the experiments, execute
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