import importlib
from copy import deepcopy
from os import path as osp
from glob import glob
import torch

from utils.registry import DATASET_REGISTRY

__all__ = ['build_train_loader', 'build_valid_loader', 'build_test_loader']

# automatically scan and import dataset modules for registry
# scan all the files under the 'datasets' folder and collect files ending with '_dataset.py'
dataset_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in glob(osp.join(dataset_folder, '*_dataset.py'))]
# import all the dataset modules
_dataset_modules = [importlib.import_module(f'datasets.{file_name}') for file_name in dataset_filenames]


def build_dataset(dataset_cfg, split: str):
    assert split.upper() in ['TRAIN', 'VALID', 'TEST']
    dataset_cfg = deepcopy(dataset_cfg)
    dataset_type = dataset_cfg.pop('type')
    process_cfg = dataset_cfg.pop('process')
    split_cfg = dataset_cfg.pop(split)
    dataset = DATASET_REGISTRY.get(dataset_type)(
        **dataset_cfg,
        **process_cfg,
        **split_cfg,
        split=split
    )
    return dataset

def build_train_loader(dataset_cfg):
    train_dataset = build_dataset(dataset_cfg, 'train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=dataset_cfg['train']['batch_size'],
                                                   shuffle=True, num_workers=dataset_cfg['num_workers'],
                                                   pin_memory=dataset_cfg['pin_memory'], persistent_workers=dataset_cfg['persistent_workers'])
    return train_dataloader

def build_valid_loader(dataset_cfg, num_workers=None):
    valid_dataset = build_dataset(dataset_cfg, 'valid')
    if num_workers is None:
        num_workers = dataset_cfg['num_workers']
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=dataset_cfg['valid']['batch_size'],
                                                   shuffle=False, num_workers=num_workers,
                                                   pin_memory=dataset_cfg['pin_memory'], persistent_workers=dataset_cfg['persistent_workers'])
    return valid_dataloader


def build_test_loader(dataset_cfg, num_workers=None):
    test_dataset = build_dataset(dataset_cfg, 'test')
    if num_workers is None:
        num_workers = dataset_cfg['num_workers']
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=dataset_cfg['test']['batch_size'],
                                                   shuffle=False, num_workers=num_workers,
                                                   pin_memory=dataset_cfg['pin_memory'], persistent_workers=dataset_cfg['persistent_workers'])
    return test_dataloader