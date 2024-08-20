#!/bin/bash
set -e

alphas=0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
for alpha in $(seq 0.0 1.0 0.1); do
    for sample in $(seq 1 4 1); do
        python3 -m main.py \
            --dataset ZINC \
            --alpha $alpha \
            --gnn GCN \
            --gnn_layer_sizes '8*11' \
            --task Graph-R \
            --dropout DropEdge \
            --drop_p $1 \
            --n_epochs 500 \
            --learning_rate 0.001 \
            --weight_decay 0.0005 \
            --test_every -1
    done
done