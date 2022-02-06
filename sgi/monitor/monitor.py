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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ..util import numel
from ..data import MetadataCatalog
from ..data import build_clevr_dataset as build_dataset
from ..data import process_clevr_batch as process_batch
from ..data import build_copy_dataset, process_copy_batch
from ..data import build_clevr_dataset, process_clevr_batch
from ..model import build_main_model
from ..module import LARS, exclude_bias_or_norm, adjust_learning_rate

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
        self.build_optimizer(tunable_params)

    def show_batch(self, batch, meta):
        def recover_boxes(boxes, width, height):
            boxes[:, 0::2] *= width 
            boxes[:, 1::2] *= height 
            boxes[:, 2] -= boxes[:, 0] 
            boxes[:, 3] -= boxes[:, 1] 
            boxes = boxes.round().astype(int).tolist()
            return boxes
            
        for k, (boxes, names, classes, caption, width, height) in enumerate(zip(
                batch["obj_boxes"], batch["obj_names"], batch["obj_classes"], 
                batch["caption"], batch["width"], batch["height"]
            )):
            if isinstance(boxes, Tensor): 
                # (x, y, w, h)
                boxes = boxes.detach().clone()
                boxes = recover_boxes(boxes.numpy(), width, height)
            elif isinstance(boxes, list):
                boxes = np.array(boxes)
                if boxes.ndim == 2:
                    boxes = recover_boxes(boxes, width, height)
                else:
                    boxes = boxes.tolist()

            names  = names.tolist() if isinstance(names, Tensor) else names 
            classes = classes.tolist() if isinstance(classes, Tensor) else classes 
            caption = caption.tolist() if isinstance(caption, Tensor) else caption 

            # object names 
            names = [self.encoder_vocab(name) for name in names]
            names = [" ".join(name) if isinstance(name, list) else name for name in names]
            # class id -> class name
            classes = [meta.thing_classes[k] for k in classes]
            # captions
            caption = " | ".join(self.decoder_vocab(caption))
            self.echo(
                f"\nbox: {boxes}\nname: {names}\nclass: {classes}\ncaption: {caption}"
            )

    def build_data(self):
        # `cate_vocab` consists of combinations of attribute words
        self.dataloader, self.evalloader, self.testloader, \
        self.encoder_vocab, self.decoder_vocab, self.cate_vocab = \
            eval(f"build_{self.cfg.data.name}_dataset")(
                self.cfg.data, not self.cfg.eval, self.echo
            )
        """
        meta = MetadataCatalog.get(self.cfg.data.name)
        for epoch_step, batch_data in enumerate(self.dataloader):
            self.show_batch(batch_data, meta)
            print(batch_data)
            break
        import sys; sys.exit(0)
        """

    def make_batch(self, union):
        obj_idxes, obj_boxes, obj_masks, sequences, obj_names, obj_name_lengths, img_shape = \
            eval(f"process_{self.cfg.data.name}_batch")(
                union, self.encoder_vocab, self.decoder_vocab, self.cate_vocab, self.device, 
                max_num_obj=self.cfg.data.max_num_obj
            )
        obj_names = self.model.backbone(obj_names, obj_name_lengths)
        return {
            "obj_idxes": obj_idxes, "obj_boxes": obj_boxes, "obj_names": obj_names, 
            "obj_masks": obj_masks, "sequences": sequences, "img_shape": img_shape,
        }

    def epoch(self, iepoch):
        self.model.reset()
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
            if epoch_step < 9544:
                pass #continue
            ppl_criteria = do_batch(batch_data, epoch_step + 1)

        if not self.cfg.optimizer.use_lars and not self.cfg.optimizer.batch_sch and \
            self.scheduler is not None:
            self.scheduler.step()
        self.timeit(all_time, show=True)
        return ppl_criteria 

    def infer(self, dataloader, samples=float("inf"), iepoch=0):
        losses, istep, nsample, nchunk, nbatch = 0, 0, 0, 1, len(dataloader)
        device_ids = [i for i in range(self.cfg.num_gpus)]
        if isinstance(self.model, DistributedDataParallel):
            dataloader.sampler.set_epoch(iepoch)
            nchunk = self.cfg.num_gpus
        peep_rate = max(10, (len(dataloader) // 10))
        start_time = time.time()
        epoch_step = total_word = 0
        loss_per_length = defaultdict(list)
        meta = MetadataCatalog.get(self.cfg.data.name)
        for ibatch, batch in enumerate(dataloader):
            if nsample >= samples:
                #print(f"{nsample}\t{ibatch}/{nbatch} continue")
                break #continue # iterate through every batch 

            #self.show_batch(batch, meta)

            batch_dict = self.make_batch(batch)
            sequences = batch_dict["sequences"]

            _, (_, loss_out) = self.model(**batch_dict)
            ntoken, loss_all = loss_out 
            loss = loss_all.sum()

            total_word += ntoken
            epoch_step += 1

            L = sequences.shape[-1] - 2 
            if L not in loss_per_length:
                loss_per_length[L] = [0., 0.]
            loss_and_count = loss_per_length[L]
            loss_and_count[0] += loss
            loss_and_count[1] += ntoken 

            nsample += sequences.shape[0] * nchunk
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
        result = " ".join([f"{k}: {v[0] / v[1]:.2E}" for k, v in loss_per_length.items()])
        self.echo(f"Length-Loss - {result}")
        return model.report()
        
    def infer_placeholder(self, dataloader, samples=float("inf"), iepoch=0):
        total_loss = 0.
        total_sent = 0.
        start_time = time.time()
        ll_total, num_token = 0., 0.
        tp_rel_word, p_rel_word = 0., 0.
        tp_rel, p_rel = 0., 0.
        tp, p = 0., 0.

        relations = list() 
        gen_h_all = defaultdict(list)
        rel_h_all = defaultdict(list)
        gen_p_all = defaultdict(list)
        attn_mean = defaultdict(list)
        sign_mean = defaultdict(list)
        placeholder = (relations, gen_h_all, rel_h_all, gen_p_all, attn_mean, sign_mean)

        rel_words = ["left", "right", "front", "behind"]
        rel_word_ids = self.decoder_vocab(rel_words)
        rel_accuracy = {w: [0] * 3 for w in rel_words} 

        def print_eval_info():
            v = ll_total / num_token
            self.echo(f"Word-level PPL: {np.exp(v):.4f} = exp({v:.4f}) = exp({ll_total:.4f} / {num_token})")
            if True or self.cate_vocab is not None:
                self.echo(
                    f"Relation acc: {(tp_rel / p_rel):.4f} = {tp_rel:.4f} / {p_rel}) " + 
                    f"Rel word acc: {(tp_rel_word / p_rel_word):.4f} = {tp_rel_word:.4f} / {p_rel_word}) Overall acc: {(tp / p):.4f}"
                )
                msg = ""
                for k, v in rel_accuracy.items():
                    msg += f"{k}: {v[0] / max(v[1], 1):.4f} = {v[0]} / {v[1]} "
                self.echo(msg)
    
        def per_rel_accuracy(sequences, predictions):
            for idx, word_id in enumerate(rel_word_ids):
                rel_mask = (sequences != -100).fill_(False)
                rel_mask |= (sequences == word_id) 
                key_word = rel_words[idx]

                rel_correct = ((predictions == sequences) * rel_mask.float()).sum()
                rel_word_count = rel_mask.sum()

                rel_accuracy[key_word][0] += rel_correct
                rel_accuracy[key_word][1] += rel_word_count

        def relation_accuracy(sequences, logits, batch, key="new_caption"):
            sequences = [sequence[1:-1] for sequence in batch[key]]
            sequences = pad_sequence(
                sequences, batch_first=True, padding_value=self.decoder_vocab.PAD_IDX
            ).cuda()
            mask = sequences != self.decoder_vocab.PAD_IDX

            predictions = logits.argmax(-1) # overgeneration
            predictions = predictions[:, :sequences.size(1)]

            all_correct = ((predictions == sequences) * mask.float()).sum()
            all_count = mask.sum()
            
            # evaluate at relation word positions
            rel_mask = (sequences != -100).fill_(False)
            for word_id in rel_word_ids:
                rel_mask |= (sequences == word_id) 
            rel_correct = ((predictions == sequences) * rel_mask.float()).sum()
            rel_word_count = rel_mask.sum()

            per_rel_accuracy(sequences, predictions)
            
            # select relation positions
            name = "" if self.cate_vocab is None else self.cate_vocab.name
            if name == "atomic_object":
                predictions = predictions[:, 2::3]
                sequences = sequences[:, 2::3]
                mask = mask[:, 2::3]
            elif "_oor" in name:
                predictions = predictions[:, 8::9]
                sequences = sequences[:, 8::9]
                mask = mask[:, 8::9]
            elif "_oro" in name:
                predictions = predictions[:, 4::9]
                sequences = sequences[:, 4::9]
                mask = mask[:, 4::9]

            correct = ((predictions == sequences) * mask.float()).sum()
            rel_count = mask.sum()
            return correct, rel_count, all_correct, all_count, rel_correct, rel_word_count

        for idx, batch in enumerate(dataloader):
            batch_dict = self.make_batch(batch) 

            optim_step += 1 
            bsize = batch_dict["sequences"].shape[0]

            loss, (rels, outs) = self.model(**batch_dict)

            (ll, ntoken, logits, attns) = outs[:4]

            ll_total += ll.item()
            num_token += ntoken.item()

            k = 1 # logits
            inter_attn = attns[-1][1][k]
            len_src = inter_attn.size(-1)
            if len(attns[-1][1]) > 2: # attns' signs
                inter_sign = attns[-1][1][2]
                if inter_sign is not None:
                    inter_attn = (inter_attn, inter_sign)
            outputs = (rels, outs[4] if len(outs) == 5 else logits, inter_attn) 
            key = "caption" if self.cate_vocab is None else "new_caption"
            is_pair = len_src == obj_idxes.size(-1) ** 2
            summary_outputs(outputs, union, *placeholder, key=key, raw_rel=True, raw_gen=True, pair=is_pair) 

            if True or self.cate_vocab is not None:
                correct, rel_count, all_correct, all_count, rel_correct, rel_word_count = \
                    relation_accuracy(sequences, logits, union, key=key)
                tp += all_correct
                p += all_count
                tp_rel += correct
                p_rel += rel_count
                tp_rel_word += rel_correct
                p_rel_word += rel_word_count

            iepoch = 0
            optim_step += 1
            total_loss += loss.item()
            total_sent += sequences.size(0) 
            if self.optim_step % self.cfg.peep_every == 0:
                self.echo(
                    f"epoch {iepoch}\tstep {self.optim_step}\t" + 
                    f"loss {total_loss / self.optim_step:.6f}\t" + 
                    f"{total_sent / (time.time() - start_time):.2f} samples/sec"
                )
                if self.optim_step % (self.cfg.peep_every * 1) == 0:
                    print_eval_info()

            if self.cfg.save_eval and idx >= 500:
                break
        print_eval_info()

        if not self.cfg.save_eval:
            return 
        
        """
        attn_mean = (attn_mean, sign_mean) if len(sign_mean) > 0 else attn_mean
        data = ((gen_h_all, attn_mean), rel_h_all, relations, ll_total/ num_token)
        mfile = ".".join(self.cfg.model_file.split("/")[-2:])
        dname = "train" if "train" in self.cfg.data_name else "val"
        #ofile = f"/home/s1847450/data/scenesave/stats/{mfile}-{dname}"
        ofile = f"/home/s1847450/data/scenesave/stats/logits-{mfile}-{dname}"
        self.echo(f"saving to {ofile}")
        torch.save(data, ofile)
        """
