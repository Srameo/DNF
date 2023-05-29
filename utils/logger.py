# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import sys
import logging
import functools
from termcolor import colored

logger = None

@functools.lru_cache()
def create_logger(output_dir, dist_rank=None, name='', action='train'):
    global logger
    if logger is not None:
        return logger
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    if dist_rank is not None:
        # create console handlers for master process
        if dist_rank == 0:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(
                logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
            logger.addHandler(console_handler)

        file_handler = logging.FileHandler(os.path.join(output_dir, f'{action}-rank-{dist_rank}.log'), mode='a')
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)
        file_handler = logging.FileHandler(os.path.join(output_dir, f'{action}.log'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger
