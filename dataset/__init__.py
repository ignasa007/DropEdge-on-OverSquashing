from dataset.base import BaseDataset
from dataset.planetoid import Cora, CiteSeer, PubMed
from dataset.qm9 import QM9
from dataset.tudataset import Proteins, PTC, MUTAG
from dataset.lrgb import Pascal
from dataset.synthetic_zinc import SyntheticZINC
from dataset.synthetic_mutag import SyntheticMUTAG
from dataset.twitch import TwitchDE
from dataset.actor import Actor
from dataset.wikipedia import Chameleon, Crocodile, Squirrel


def get_dataset(dataset_name: str, **kwargs) -> BaseDataset:

    dataset_map = {
        'cora': Cora,
        'citeseer': CiteSeer,
        'pubmed': PubMed,
        'qm9': QM9,
        'proteins': Proteins,
        'ptc': PTC,
        'mutag': MUTAG,
        'pascal': Pascal,
        'syntheticzinc': SyntheticZINC,
        'syntheticmutag': SyntheticMUTAG,
        'twitchde': TwitchDE,
        'actor': Actor,
        'chameleon': Chameleon,
        'crocodile': Crocodile,
        'squirrel': Squirrel,
    }
    
    formatted_name = dataset_name.lower()
    if formatted_name not in dataset_map:
        raise ValueError(f'Parameter `dataset_name` not recognised (got `{dataset_name}`).')
    
    dataset = dataset_map.get(formatted_name)
    
    return dataset(**kwargs)