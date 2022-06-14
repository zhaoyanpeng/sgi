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

from ..util import numel, cat_name_list, estimate_precision_from_dict, update_precision_from_dict
from ..data import MetadataCatalog, mask_tokens, connect_class2token
from ..data import build_clevr_dataset as build_dataset
from ..data import process_clevr_batch as process_batch
from ..data import build_copy_dataset, process_copy_batch
from ..data import build_clevr_dataset, process_clevr_batch
from ..data.clevr_constant import type_attrs_ext, attr_types, syn_attrs
from ..model import build_main_model
from ..module import LARS, exclude_bias_or_norm, adjust_learning_rate

from .monitor import Monitor

class Monitor(Monitor):
    def __init__(self, cfg, echo, device):
        super(Monitor, self).__init__(cfg, echo, device)

    def build_data(self):
        # `cate_vocab` consists of combinations of attribute words
        self.dataloader, self.evalloader, self.testloader, \
        self.encoder_vocab, self.decoder_vocab, self.cate_vocab = \
            eval(f"build_{self.cfg.data.name}_dataset")(
                self.cfg.data, not self.cfg.eval, self.echo
            )
        self.token2class = {}
        if self.cfg.data.cate_type in {"atomic_object"}:
            meta = MetadataCatalog.get(self.cfg.data.name)
            self.token2class = connect_class2token(self.decoder_vocab, meta)
        """
        meta = MetadataCatalog.get(self.cfg.data.name)
        for epoch_step, batch_data in enumerate(self.dataloader):
            self.show_batch(batch_data, meta)
            print(batch_data)
            break
        import sys; sys.exit(0)
        """

    def make_batch(self, union):
        obj_idxes, obj_boxes, obj_masks, sequences, obj_names, obj_name_lengths, img_shape, objects = \
            eval(f"process_{self.cfg.data.name}_batch")(
                union, self.encoder_vocab, self.decoder_vocab, self.cate_vocab, self.device, 
                max_num_obj=self.cfg.data.max_num_obj
            )
        obj_names_src = obj_names
        file_name = union["file_name"]
        obj_names = objects #self.model.backbone(obj_names, obj_name_lengths)

        mlm_inputs = mlm_labels = None
        mlm_prob=self.cfg.data.mlm_prob
        at_least_one = self.cfg.data.cate_type != "" # TODO
        if mlm_prob > 0:
            mlm_inputs, mlm_labels = mask_tokens(
                sequences, mlm_prob, self.decoder_vocab, train=self.model.training, #False, #
                target_words=list(self.cfg.data.relation_words), at_least_one=at_least_one,
            )
        inter_attn_mask = None
        if False and len(self.token2class) > 0:
            if self.cfg.data.cate_type == "atomic_object":
                B, T = sequences.shape
                device = sequences.device
                tokens = sequences[:, 1::2].cpu().tolist()
                gold_obj_idxes = torch.tensor([
                    [obj_list.index(self.token2class[token]) for token in token_list]
                    for token_list, obj_list in zip(tokens, union["obj_classes"])
                ], device=device)
                inter_attn_mask = torch.full(obj_idxes.shape, 1, device=device).bool()
                inter_attn_mask.scatter_(-1, gold_obj_idxes, 0)

                """
                indice = torch.arange(B, device=device)
                inter_attn_mask = inter_attn_mask.unsqueeze(1).repeat(1, T, 1)
                inter_attn_mask[indice, 1, gold_obj_idxes[:, 1]] = True
                inter_attn_mask[indice, 3, gold_obj_idxes[:, 0]] = True
                """

                #print(gold_obj_idxes)

            elif "_oro" in self.cfg.data.cate_type:
                pass
            elif "_oor" in self.cfg.data.cate_type:
                pass
        return {
            "obj_idxes": obj_idxes, "obj_boxes": obj_boxes, "obj_names": obj_names, 
            "obj_masks": obj_masks, "sequences": sequences, "img_shape": img_shape,
            "file_name": file_name, "obj_names_src": obj_names_src,
            "mlm_inputs": mlm_inputs, "mlm_labels": mlm_labels,
            "inter_attn_mask": inter_attn_mask,
        }

    def epoch(self, iepoch):
        self.model.reset()
        epoch_beta = self.epoch_beta[iepoch]
        all_time = defaultdict(list)
        self.timeit(all_time)        
        device_ids = [i for i in range(self.cfg.num_gpus)]
        nchunk = dist.get_world_size() if torch.distributed.is_initialized() else 1  
        warmup_step_rate = max(self.cfg.optimizer.warmup_steps // 20, 1)
        meta = MetadataCatalog.get(self.cfg.data.name)
        #for step, batch in enumerate(self.dataloader, start=iepoch * len(self.dataloader)):
        def do_batch(batch, step):
            ppl_criteria = -1 # FIXME local variable will not override the global one
            epoch_step = step #% len(self.dataloader)
            
            #self.show_batch(batch, meta)

            batch_dict = self.make_batch(batch) 
            batch_dict["epoch_beta"] = epoch_beta

            self.optim_step += 1 
            bsize = batch_dict["sequences"].shape[0]
            force_eval, warmup = self.pre_step(step, warmup_step_rate)

            self.timeit(all_time, key="data")

            loss, _ = self.model(**batch_dict)
            loss.backward()
            if self.optim_step % self.cfg.running.optim_rate == 0: 
                self.step()

            self.timeit(all_time, key="model")

            self.num_sents += bsize 

            self.total_step += 1
            self.epoch_step += 1
            self.total_loss += loss.detach()
            self.epoch_loss += loss.detach()
            self.total_inst += bsize * nchunk

            ppl_criteria = self.post_step(
                iepoch, epoch_step, force_eval, warmup, nchunk, 
            )

            self.timeit(all_time, key="report")
            return ppl_criteria

        for epoch_step, batch_data in enumerate(self.dataloader):
            if iepoch < 38:
                pass #continue
            if epoch_step < 600:
                pass #continue
            ppl_criteria = do_batch(batch_data, epoch_step + 1)

        if not self.cfg.optimizer.use_lars and not self.cfg.optimizer.batch_sch and \
            self.scheduler is not None:
            self.scheduler.step()
        self.timeit(all_time, show=True)

        self.total_step = self.total_loss = self.total_inst = 0
        self.start_time = time.time()

        return ppl_criteria 

