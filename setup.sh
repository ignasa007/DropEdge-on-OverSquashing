#!/bin/sh
set -e

pip install -r requirements.txt

conda install pytorch==1.13.0 cpuonly -c pytorch
pip install torchmetrics==1.4.0 lightning-utilities==0.11.2

conda install pyg -c pyg
pip install pyg_lib==0.4.0 torch_cluster==1.6.1 torch_scatter==2.1.1 torch-sparse==0.6.17 torch_spline_conv==1.2.2 -f https://pytorch-geometric.com/whl/torch-1.13.0+cpu.html