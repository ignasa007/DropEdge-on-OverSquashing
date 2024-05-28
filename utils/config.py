from yacs.config import CfgNode as CN


def default_cfg():

    '''
    The default configuration object for the experiments.
        - Need to register all configurations that are expected.

    Return:
        _C: A configuration object with placeholder values.
    '''

    _C = CN()

    _C.device_index = None

    _C.add_self_loops = None
    _C.normalize = None
    
    _C.h_layer_sizes = None
    _C.dropout_prob = None
    _C.activation = None
    
    _C.lr = None
    _C.n_epochs = None

    return _C.clone()


class Config:
    
    def __init__(self, root, override=None):

        '''
        Initialization of the configuration object used by the main file.

        Args:
            root (str): file path for the default configurations.
            dataset (str): file path for the dataset configurations used in
                the experiment.
            model (str): file path for the model configurations used in the
                experiment.
            override (Union[list, None]): key-value pairs with command-line
                arguments indicating the configurations to override.
        '''

        self.cfg = default_cfg()
        self.cfg.merge_from_file(f'{root}/config.yaml')

        if isinstance(override, list):
            self.cfg.merge_from_list(override)

    def __getattr__(self, name: str):

        '''
        Method for returning configurations using dot operator.
        '''

        return self.cfg.__getattr__(name)