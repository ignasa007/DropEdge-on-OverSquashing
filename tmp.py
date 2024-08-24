import os
import pickle
import torch
from model import Model


with open('./results/ct-triu/GCN/L=11/P=0.0/alpha=0.0/2024-08-23-03-57-53/config.pkl', 'rb') as f:
    config = pickle.load(f)

for sample in range(11, 12):
    model = Model(config)
    sample_fn = f'./results/state-dicts/sample-{sample}.pt'
    os.makedirs(os.path.dirname(sample_fn), exist_ok=True)
    torch.save(model.state_dict(), sample_fn)
