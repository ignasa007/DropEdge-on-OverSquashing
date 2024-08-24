#!/bin/bash
set -e

for alpha in $(seq 0.0 0.1 1.0); do
    for sample in $(seq 11 1 11); do
        python3 -m main \
            --dataset ZINC \
            --alpha $alpha \
            --gnn GCN \
            --gnn_layer_sizes '8*11' \
            --task Graph-R \
            --dropout DropEdge \
            --drop_p $1 \
            --n_epochs 250 \
            --learning_rate 0.002 \
            --weight_decay 0.0001 \
            --test_every -1 \
	    --model_sample $sample;
    done
done
