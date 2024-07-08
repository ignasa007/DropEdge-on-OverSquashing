import os
from argparse import Namespace
from datetime import datetime
import pickle
from typing import List, Tuple

import numpy as np
import torch


def get_time():
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


class Logger:

    def __init__(self, config: Namespace):

        '''
        Initialize the logging directory:
            ./results/<dataset>/<gnn_layer>/<drop_strategy>/<datetime>/

        Args:
            dataset (str): dataset name.
            model (str): model name.
        '''
        
        self.exp_dir = f'./results/drop-edge/long-range/{get_time()}'; os.makedirs(self.exp_dir)
        self.log(''.join(f'{k} = {v}\n' for k, v in vars(config).items()), with_time=False)
        with open(f'{self.exp_dir}/config.pkl', 'wb') as f:
            pickle.dump(config, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.pickle_dir = None
        self.array_dir = None
        self.tensor_dir = None

    def log(
        self,
        text: str,
        with_time: bool = True,
        print_text: bool = False,
    ):

        '''
        Write logs to the the logging file: ./<exp_dir>/logs

        Args:
            text (str): text to write to the log file.
            with_time (bool): prepend text with datetime of writing.
            print_text (bool): print the text to console, in addition
                to writing it to the log file.
        '''

        if print_text:
            print(text)
        if with_time:
            text = f"{get_time()}: {text}"
        with open(f'{self.exp_dir}/logs', 'a') as f:
            f.write(text + '\n')

    def log_metrics(
        self,
        metrics: List[Tuple[str, float]],
        prefix: str = '',
        with_time: bool = True,
        print_text: bool = False
    ):

        formatted_metrics = prefix
        formatted_metrics += ', '.join(f'{metric} = {value:.6e}' for metric, value in metrics)
        self.log(formatted_metrics, with_time, print_text)

    def save_pickle(self, fn, obj):

        '''
        Save a Python object as a (binary) pickle file.

        Args:
            fn (str): file name to save the object at.
            obj (Any): Python object to save.
        '''

        if self.pickle_dir is None:
            self.pickle_dir = f'{self.exp_dir}/pickle'; os.makedirs(self.pickle_dir)

        if not fn.endswith('.pkl'):
            fn = os.path.splitext(fn)[0] + '.pkl'
        with open(f'{self.pickle_dir}/{fn}', 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_arrays(self, fn, *config, **kwconfig):

        '''
        Save NumPy arrays.

        Args:
            config (List[np.ndarray]): arrays saved into <fn>_unnamed.npz 
                and can be queried using integer indexing.
            kwconfig (Dict[str, np.ndarray]): arrays saved into <fn>_named.npz 
                and can be queried using string indexing.
        '''

        if self.array_dir is None:
            self.array_dir = f'{self.exp_dir}/arrays'; os.makedirs(self.array_dir)

        if config:
            assert all([isinstance(ar, np.ndarray) for ar in config]), \
                f'Expected NumPy arrays, instead received {set((type(ar) for ar in config))}.'
            unnamed_fn = os.path.splitext(fn)[0] + '_unnamed.npz'
            with open(f'{self.array_dir}/{unnamed_fn}', 'wb') as f:
                np.savez(f, *config)
        if kwconfig:
            assert all([isinstance(ar, np.ndarray) for ar in kwconfig.keys()]), \
                f'Expected NumPy arrays, instead received {set((type(ar) for ar in kwconfig.keys()))}.'
            named_fn = os.path.splitext(fn)[0] + '_named.npz'
            with open(f'{self.array_dir}/{named_fn}', 'wb') as f:
                np.savez(f, **kwconfig)

    def save_tensors(self, fn, tensor):

        '''
        Save a PyTorch tensor object.

        Args:
            fn (str): file name to save the tensor at.
            obj (torch.Tensor): Torch tensor to save.
        '''

        if self.tensor_dir is None:
            self.tensor_dir = f'{self.exp_dir}/tensors'; os.makedirs(self.tensor_dir)

        assert isinstance(tensor, torch.Tensor), \
            f'Expected Torch tensor, instead received {type(tensor)}.'
        if not fn.endswith('.pt'):
            fn = os.path.splitext(fn)[0] + '.pt'
        torch.save(tensor, f'{self.tensor_dir}/fn')