import argparse
import os
import random
import torch
import yaml
from collections import OrderedDict
from os import path as osp
import numpy as np
from copy import deepcopy


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        tuple: yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def yaml_load(f):
    """Load yaml file or string.

    Args:
        f (str): File path or a python string.

    Returns:
        dict: Loaded dict.
    """
    if os.path.isfile(f):
        with open(f, 'r') as f:
            return yaml.load(f, Loader=ordered_yaml()[0])
    else:
        return yaml.load(f, Loader=ordered_yaml()[0])


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg


def _postprocess_yml_value(value):
    # None
    if value == '~' or value.lower() == 'none':
        return None
    # bool
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    # !!float number
    if value.startswith('!!float'):
        return float(value.replace('!!float', ''))
    # number
    if value.isdigit():
        return int(value)
    elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
        return float(value)
    # list
    if value.startswith('['):
        return eval(value)
    # str
    return value


def merge_from_base(cfg, cfg_path):
    def _merge_a_into_b(cfg_a, cfg_b):
        for k, v_ in cfg_a.items():
            v = deepcopy(v_)
            if isinstance(v, dict) and k in cfg_b:
                _merge_a_into_b(v, cfg_b[k])
            else:
                cfg_b[k] = v

    if 'base' in cfg:
        if isinstance(cfg['base'], str):
            cfg['base'] = [ cfg['base'] ]
        for base_cfg_path in cfg['base']:
            full_base_cfg_path = os.path.join(os.path.dirname(cfg_path), base_cfg_path)
            base_cfg = yaml_load(full_base_cfg_path)
            base_cfg = merge_from_base(base_cfg, full_base_cfg_path)
            _merge_a_into_b(cfg, base_cfg)
            
        return base_cfg
    return cfg
            
            
def set_default_config(cfg):
    if 'train' not in cfg:
        cfg['train'] = {}
    
    ##### default #####
    if 'output' not in cfg:
        cfg['output'] = 'runs'
    if 'tag' not in cfg:
        cfg['tag'] = 'debug'
    
    ##### data #####
    if 'persistent_workers' not in cfg['data']:
        cfg['data']['persistent_workers'] = False
    if 'train' in cfg['data'] and 'repeat' not in cfg['data']['train']:
        cfg['data']['train']['repeat'] = 1
    # augmentation 
    if 'transpose' not in cfg['data']['process']:
        cfg['data']['process']['transpose'] = False
    if 'h_flip' not in cfg['data']['process']:
        cfg['data']['process']['h_flip'] = True
    if 'v_flip' not in cfg['data']['process']:
        cfg['data']['process']['v_flip'] = True
    if 'rotation' not in cfg['data']['process']:
        cfg['data']['process']['rotation'] = False
        
    ##### train #####
    if 'auto_resume' not in cfg['train']:
        cfg['train']['auto_resume'] = False
        
    ##### test #####
    if 'test' in cfg:
        if 'round' not in cfg['test']:
            cfg['test']['round'] = False
        if 'save_image' not in cfg['test']:
            cfg['test']['save_image'] = False
        
    cfg['output'] = os.path.join(cfg.get('output', 'runs'), cfg['name'], cfg.get('tag', ''))


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--auto-resume', action='store_true', default=False, help='Auto resume from latest checkpoint')
    parser.add_argument('--resume', type=str, default=None, help='Path to resume.')
    parser.add_argument('--pretrain', type=str, default=None, help='Path to the pretrained checkpoint path.')
    parser.add_argument('--test', action='store_true', default=False, help='Test mode')
    parser.add_argument('--save-image', action='store_true', default=False, help='Save image during test or validation')
    parser.add_argument(
        '--force-yml', nargs='+', default=None, help='Force to update yml files. Examples: train:ema_decay=0.999')
    args = parser.parse_args()

    # parse yml to dict
    cfg = yaml_load(args.cfg)
    cfg = merge_from_base(cfg, args.cfg)
    set_default_config(cfg)

    # random seed
    seed = cfg.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        cfg['manual_seed'] = seed
    set_random_seed(seed)
    
    if args.test:
        assert not args.auto_resume and \
            (args.resume is not None or cfg['train'].get('resume') is not None) or \
            (args.pretrain is not None or cfg['train'].get('pretrained') is not None)
        cfg['testset_as_validset'] = True
        cfg['eval_mode'] = True
    
    if args.auto_resume:
        assert args.resume is None
        cfg['train']['auto_resume'] = True

    if args.resume:
        assert args.pretrain is None
        cfg['train']['resume'] = args.resume
        
    if args.pretrain:
        cfg['train']['pretrained'] = args.pretrain

    if args.save_image:
        cfg['test']['save_image'] = True

    # force to update yml options
    if args.force_yml is not None:
        for entry in args.force_yml:
            # now do not support creating new keys
            keys, value = entry.split('=')
            keys, value = keys.strip(), value.strip()
            value = _postprocess_yml_value(value)
            eval_str = 'cfg'
            for key in keys.split(':'):
                eval_str += f'["{key}"]'
            eval_str += '=value'
            # using exec function
            exec(eval_str)

    return args, cfg


def copy_cfg(cfg, filename):
    # copy the yml file to the experiment root
    import sys
    import time
    from shutil import copyfile
    cmd = ' '.join(sys.argv)

    with open(filename, 'w') as f:
        lines = [f'# GENERATE TIME: {time.asctime()}\n# CMD:\n# {cmd}\n\n']
        lines.append(yaml.dump(ordered_dict_to_dict(cfg), default_flow_style=False, sort_keys=False))
        f.writelines(lines)


def ordered_dict_to_dict(cfg):
    cfg_dict = {}
    for k, v in cfg.items():
        if isinstance(v, OrderedDict):
            cfg_dict[k] = ordered_dict_to_dict(deepcopy(v))
        else:
            cfg_dict[k] = v
    return cfg_dict
