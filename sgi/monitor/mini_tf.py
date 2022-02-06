from omegaconf import OmegaConf
import os, re
from collections import defaultdict

import json
import time
import torch
import random
import numpy as np
from torch import nn, Tensor

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from torch.utils.data import dataset
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from .monitor import Monitor

class Monitor(Monitor):
    def __init__(self, cfg, echo, device):
        super(Monitor, self).__init__(cfg, echo, device)

    def show_batch(self, batch, meta):
        pass

    def build_data(self):
        train_iter = WikiText2(split='train')
        tokenizer = get_tokenizer('basic_english')
        vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
        vocab.set_default_index(vocab['<unk>'])

        def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
            """Converts raw text into a flat Tensor."""
            data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
            return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

        # train_iter was "consumed" by the process of building the vocab,
        # so we have to create it again
        train_iter, val_iter, test_iter = WikiText2()
        train_data = data_process(train_iter)
        val_data = data_process(val_iter)
        test_data = data_process(test_iter)

        def batchify(data: Tensor, bsz: int) -> Tensor:
            """Divides the data into bsz separate sequences, removing extra elements
            that wouldn't cleanly fit.

            Args:
                data: Tensor, shape [N]
                bsz: int, batch size

            Returns:
                Tensor of shape [N // bsz, bsz]
            """
            seq_len = data.size(0) // bsz
            data = data[:seq_len * bsz]
            data = data.view(bsz, seq_len).t().contiguous()
            return data.to(self.device)

        batch_size = 20
        eval_batch_size = 10
        self.dataloader = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
        self.evalloader = batchify(val_data, eval_batch_size)
        self.testloader = batchify(test_data, eval_batch_size)

        self.encoder_vocab = self.decoder_vocab = self.vocab = vocab

        #for epoch_step, batch_data in enumerate(self.dataloader):
        #    self.show_batch(batch_data, None)
        #    #print(batch_data)
        #    break
        #import sys; sys.exit(0)

    def make_batch(self, source, i):
        """ Batch-first: (B, L) & (B x L).
        """
        seq_len = min(self.cfg.data.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len].transpose(0, 1)
        target = source[i+1:i+1+seq_len].transpose(0, 1).reshape(-1)
        return data.to(self.device), target.to(self.device)

    def pre_step(self, step, warmup_step_rate, inc=0):
        force_eval = warmup = False
        return force_eval, warmup 

    def post_step(
        self, iepoch, epoch_step, force_eval, warmup, nchunk, num_batch
    ):
        if force_eval or (self.cfg.rank == 0 and epoch_step % self.cfg.running.peep_rate == 0):
            # msg = self.model.stats()
            # self.echo(msg)
            # example output if there is any
            #
            # overall stats 
            lr_w = self.optimizer.param_groups[0]['lr']
            lr_b = self.optimizer.param_groups[1]['lr']
            self.echo(
                f"epoch {iepoch:>4} step {epoch_step} / {num_batch}\t" + 
                f"lr_w {lr_w:.2e} lr_b {lr_b:.2e} loss {self.total_loss / self.total_step:.3f} ({self.total_step}) " + 
                f"{self.total_inst / (time.time() - self.start_time):.2f} samples/s" 
            )
            self.total_step = self.total_loss = self.total_inst = 0
            self.start_time = time.time()

        ppl_criteria = -1
        if force_eval or (self.total_step > 0 and self.total_step % self.cfg.running.save_rate == 0) or (
                self.cfg.running.save_epoch and epoch_step % num_batch == 0
            ): # distributed eval
            report = ""
            if self.evalloader is not None:
                self.model.train(False)
                with torch.no_grad():
                    report = self.infer(
                        self.evalloader, samples=self.cfg.data.eval_samples, iepoch=iepoch
                    )
                self.model.train(True)
            if report != "":
                self.echo(f"EVAL {report}")

            report = ""
            if self.testloader is not None:
                self.model.train(False)
                with torch.no_grad():
                    report = self.infer(
                        self.testloader, samples=self.cfg.data.eval_samples, iepoch=iepoch
                    )
                self.model.train(True)
            if report != "":
                self.echo(f"TEST {report}")

            if self.cfg.rank == 0:
                self.save()

        # global across epochs 
        if self.optim_step % self.cfg.running.optim_rate == 0: 
            self.model.zero_grad()
        # used for initialization search
        return ppl_criteria 

    def epoch(self, iepoch):
        self.model.reset()
        all_time = defaultdict(list)
        self.timeit(all_time)        
        device_ids = [i for i in range(self.cfg.num_gpus)]
        nchunk = dist.get_world_size() if torch.distributed.is_initialized() else 1  
        warmup_step_rate = max(self.cfg.optimizer.warmup_steps // 20, 1)
        num_batch = np.ceil(len(self.dataloader) / self.cfg.data.bptt).astype(int)
        def do_batch(batch, step):
            ppl_criteria = -1 # FIXME local variable will not override the global one
            epoch_step = step #% len(self.dataloader)

            #self.show_batch(batch, meta)

            batch_dict = batch 

            self.optim_step += 1 
            bsize, length = batch_dict[0].shape
            force_eval, warmup = self.pre_step(step, warmup_step_rate)

            self.timeit(all_time, key="data")

            loss, _ = self.model(*batch_dict)
            loss.backward()
            if self.optim_step % self.cfg.running.optim_rate == 0: 
                self.step()

            self.timeit(all_time, key="model")

            self.num_sents += bsize 

            self.total_step += 1
            self.total_loss += loss.detach()
            self.total_inst += bsize * nchunk

            ppl_criteria = self.post_step(
                iepoch, epoch_step, force_eval, warmup, nchunk, num_batch,
            )

            self.timeit(all_time, key="report")
            return ppl_criteria

        for epoch_step, i in enumerate(range(0, self.dataloader.size(0) - 1, self.cfg.data.bptt)):
            batch_data = self.make_batch(self.dataloader, i)
            ppl_criteria = do_batch(batch_data, epoch_step + 1)

        if not self.cfg.optimizer.use_lars and not self.cfg.optimizer.batch_sch and \
            self.scheduler is not None:
            self.scheduler.step()
        self.timeit(all_time, show=True)
        return ppl_criteria 
        
    def infer(self, dataloader, samples=float("inf"), iepoch=0):
        nchunk = 1
        total_loss = 0.
        total_inst = 0.
        start_time = time.time()
        
        for epoch_step, i in enumerate(range(0, dataloader.size(0) - 1, self.cfg.data.bptt)):
            batch_dict = self.make_batch(dataloader, i)
            bsize, length = batch_dict[0].shape

            loss, _ = self.model(*batch_dict)
            total_loss += loss.detach() * length
            total_inst += bsize * nchunk

        loss = total_loss / (len(dataloader) - 1)
        return f"{loss:.3f}"
