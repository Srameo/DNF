import importlib
from copy import deepcopy
from os import path as osp
from glob import glob

from utils.registry import FORWARD_REGISTRY

__all__ = ['build_forwards', 'build_profile']

# automatically scan and import forward modules for registry
# scan all the files under the 'forwards' folder and collect files ending with '_forward.py'
forward_folder = osp.dirname(osp.abspath(__file__))
forward_filenames = [osp.splitext(osp.basename(v))[0] for v in glob(osp.join(forward_folder, '*_forward.py'))]
# import all the forward modules
_forward_modules = [importlib.import_module(f'forwards.{file_name}') for file_name in forward_filenames]


def build_forwards(cfg):
    cfg = deepcopy(cfg)
    train_fwd_type = cfg['train']['forward_type']
    test_fwd_type = cfg['test']['forward_type']
    train_forward = FORWARD_REGISTRY.get(train_fwd_type)
    test_forward = FORWARD_REGISTRY.get(test_fwd_type)
    return train_forward, test_forward

def build_forward(forward_type):
    return FORWARD_REGISTRY.get(forward_type)

def build_profile(cfg):
    cfg = deepcopy(cfg)
    profile = cfg.get('profile')
    if profile is None:
        return profile
    return FORWARD_REGISTRY.get(profile)
