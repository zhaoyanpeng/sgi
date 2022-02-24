import os, sys
import logging
import random
import numpy
import torch
import torch.distributed as dist

from .module import * 

def seed_all_rng(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

def setup_logger(output_dir=None, name="pcfg", rank=0, output=None, fname="train"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    if rank == 0:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)
    if os.path.exists(output_dir):
        logger.info(f'Warning: the folder {output_dir} exists.')
    else:
        logger.info(f'Creating {output_dir}')
        if rank == 0: 
            os.makedirs(output_dir)
    if torch.distributed.is_initialized():
        dist.barrier() # output dir should have been ready
    if output is not None:
        filename = os.path.join(output_dir, f'{fname}_{rank}.out')
        handler = logging.FileHandler(filename, 'w')
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def numel(model: torch.nn.Module, trainable: bool = False):
    parameters = list(model.parameters())
    if trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique) 

def detect_nan(x):
    return torch.isnan(x).any(), torch.isinf(x).any()

def enable_print():
    sys.stdout = sys.__stdout__

def disable_print():
    sys.stdout = open(os.devnull, 'w')

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
