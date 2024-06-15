#!/bin/sh
set -e

task='Node-C'
for dataset in {'Cora','CiteSeer','PubMed'}; do
    for gnn in {'GCN','GAT','APPNP'}; do
        for prob in $(seq 0.1 0.1 0.9); do
            python -B main.py \
                --dataset $dataset \
                --gnn $gnn \
                --gnn_layer_sizes 128 128 \
                --attention_heads 2 \
                --power_iter 10 \
                --teleport_p 0.1 \
                --task $task \
                --ffn_layer_sizes \
                --dropout $1 \
                --drop_p $prob \
                --n_epochs 200 \
                --learning_rate 0.01 \
                --weight_decay 0.005 \
                --test_every 1 \
                --device $2
        done    
    done
done


dataset='QM9'
task='Graph-R'
for gnn in {'GCN','GAT','APPNP'}; do
    for prob in $(seq 0.1 0.1 0.9); do
        python -B main.py \
            --dataset $dataset \
            --gnn $gnn \
            --gnn_layer_sizes 128 128 \
            --attention_heads 2 \
            --power_iter 10 \
            --teleport_p 0.1 \
            --task $task \
            --ffn_layer_sizes \
            --dropout $1 \
            --drop_p $prob \
            --n_epochs 200 \
            --learning_rate 0.01 \
            --weight_decay 0.005 \
            --test_every 1 \
            --device $2
    done    
done