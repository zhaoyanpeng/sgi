from omegaconf import OmegaConf
import os, re
from collections import defaultdict

import json
import time
import torch
import random
import itertools
import numpy as np
from torch import nn, Tensor

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ..util import numel
from ..model import build_main_model
from ..data import build_scene_dataset

from .meta import Monitor

class Monitor(Monitor):
    def __init__(self, cfg, echo, device):
        super(Monitor, self).__init__()
        self.cfg = cfg
        self.echo = echo
        self.device = device
        self.build_data()
        model = build_main_model(cfg, echo)
        tunable_params = model.build(**{
            "encoder_vocab": self.encoder_vocab, 
            "decoder_vocab": self.decoder_vocab,
        })
        self.model = DistributedDataParallel(
            model, device_ids=[cfg.rank], find_unused_parameters=True
        ) if torch.distributed.is_initialized() else model 
        self.model.train(not cfg.eval)

    def show_batch(self, batch, meta):
        pass

    def build_data(self):
        assert self.cfg.eval, f"support only eval mode (for object encoding)"
        assert self.cfg.data.batch_size == 1, f"support only batch size of 1"

        self.dataloader, self.evalloader, self.testloader, \
        self.encoder_vocab, self.decoder_vocab, self.cate_vocab = \
            build_scene_dataset(
                self.cfg.data, not self.cfg.eval, self.echo
            )
        self.token2class = {}

    def make_batch(self, union):
        return {k: v[0].to(self.device) if isinstance(v[0], torch.Tensor) else v[0] for k, v in union.items()}

    def evaluate(self, dataloader, samples=float("inf"), iepoch=0):
        self.model.reset()
        losses, istep, nsample, nchunk, nbatch = 0, 0, 0, 1, len(dataloader)
        device_ids = [i for i in range(self.cfg.num_gpus)]
        if isinstance(self.model, DistributedDataParallel):
            dataloader.sampler.set_epoch(iepoch)
            nchunk = self.cfg.num_gpus
        peep_rate = max(10, (len(dataloader) // 10))
        start_time = time.time()
        epoch_step = total_word = 0
        loss_per_length = defaultdict(list)
        for ibatch, batch in enumerate(dataloader):
            if nsample >= samples:
                #print(f"{nsample}\t{ibatch}/{nbatch} continue")
                break #continue # iterate through every batch

            #self.show_batch(batch, meta)

            batch_dict = self.make_batch(batch)
            loss, (_, features) = self.model(**batch_dict)
            
            ### save npz
            features = features.cpu().numpy()
            vfile = batch_dict["vfile"]
            vpath = vfile.rsplit("/", 1)[0]
            if not os.path.exists(vpath):
                os.makedirs(vpath)
            np.savez_compressed(vfile, v=features)
            ### save npz

            total_word += 1 
            epoch_step += 1

            nsample += 1 # bsize == 1 
            losses += loss or 0.
            if self.cfg.rank == 0 and (ibatch + 1) % peep_rate == 0:
                self.echo(
                    f"step {istep} / {ibatch}\t" + #gnorm {grad_norm():.2f} " +
                    f"loss {losses / total_word:.8f} " # (istep + 0):.8f} " # (ibatch + 1):.8f} " +
                    f"{nsample / (time.time() - start_time):.2f} samples/s"
                )
                losses = epoch_step = total_word = 0

        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        self.echo(f"# sample {nsample}; {nsample / (time.time() - start_time):.2f} samples/s")
        return model.report()

