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

        epoch_beta = []
        milestones = list(cfg.running.milestones)
        first_two = milestones[:2]
        milestones = [0] + milestones[2:]
        if len(milestones) > 1:
            start, end = first_two
            betas = np.linspace(start, end, num=len(milestones))
            stone_list = milestones #[0] + milestones
            count_list = [stone_list[i] - stone_list[i - 1] for i in range(1, len(stone_list))]
            betas = [[beta] * count for count, beta in zip(count_list, betas)]
            epoch_beta = list(itertools.chain.from_iterable(betas))
        if len(epoch_beta) < cfg.running.epochs:
            epoch_beta = epoch_beta + [0.] * (cfg.running.epochs - len(epoch_beta))
        self.epoch_beta = epoch_beta

        self.type_attrs_ext_int = {
            k: self.decoder_vocab(v) for k, v in type_attrs_ext.items()
        } # for attr_and_rel evaluation

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
        obj_idxes, obj_boxes, obj_masks, sequences, obj_names, obj_name_lengths, img_shape = \
            eval(f"process_{self.cfg.data.name}_batch")(
                union, self.encoder_vocab, self.decoder_vocab, self.cate_vocab, self.device, 
                max_num_obj=self.cfg.data.max_num_obj
            )
        obj_names_src = obj_names
        file_name = union["file_name"]
        obj_names = self.model.backbone(obj_names, obj_name_lengths)

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

    def infer(self, dataloader, samples=float("inf"), iepoch=0):
        self.model.reset()
        losses, istep, nsample, nchunk, nbatch = 0, 0, 0, 1, len(dataloader)
        device_ids = [i for i in range(self.cfg.num_gpus)]
        if isinstance(self.model, DistributedDataParallel):
            dataloader.sampler.set_epoch(iepoch)
            nchunk = self.cfg.num_gpus
        peep_rate = max(10, (len(dataloader) // 10))

        ### special eval ###
        topk_bar = 8
        verbose = True
        l_ntrue, r_ntrue, f_ntrue, b_ntrue, l_ntotal, r_ntotal, same_ntrue, same_ntotal = [0] * 8

        left_right = self.decoder_vocab(["left", "right"]) # TODO hard-coded relations
        front_behind = self.decoder_vocab(["in_front_of", "behind"])

        acc_by_attr = {k: [0, 0] for k in self.type_attrs_ext_int}
        acc_by_attr_same = {k: [0, 0] for k in self.type_attrs_ext_int}
        ### special eval ###

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

            batch_dict["infer"] = True
            batch_dict["analyze"] = True

            loss_mean, (_, loss_out) = self.model(**batch_dict)
            ntoken, loss_all = loss_out 
            loss = loss_all.sum() if isinstance(loss_all, Tensor) else loss_mean * ntoken


            ### special eval ###
            if verbose:
                (
                    l_ntrue_a, r_ntrue_a, f_ntrue_a, b_ntrue_a, l_ntotal_a, r_ntotal_a,
                    same_ntrue_a, same_ntotal_a, acc_by_attr_a, acc_by_attr_same_a, *_
                ) = self.evaluate_attr_and_rel(
                    self.type_attrs_ext_int, left_right, front_behind, topk_bar=8, verbose=False
                )

                l_ntrue += l_ntrue_a
                r_ntrue += r_ntrue_a
                f_ntrue += f_ntrue_a
                b_ntrue += b_ntrue_a
                l_ntotal += l_ntotal_a
                r_ntotal += r_ntotal_a
                same_ntrue += same_ntrue_a
                same_ntotal += same_ntotal_a

                update_precision_from_dict(acc_by_attr, acc_by_attr_a)
                update_precision_from_dict(acc_by_attr_same, acc_by_attr_same_a)
            ### special eval ###


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
        stats = model.stats()
        if stats != "": # could be empty
            self.echo(f"EVAL STATS: {model.stats()}")
        # show special eval results
        self.show_metric(
            l_ntrue, r_ntrue, f_ntrue, b_ntrue, l_ntotal, r_ntotal,
            same_ntrue, same_ntotal, acc_by_attr, acc_by_attr_same, topk_bar=topk_bar, verbose=verbose
        )
        return model.report()

    def evaluate(self, dataloader, samples=float("inf"), iepoch=0):
        self.model.reset()
        losses, istep, nsample, nchunk, nbatch = 0, 0, 0, 1, len(dataloader)
        device_ids = [i for i in range(self.cfg.num_gpus)]
        if isinstance(self.model, DistributedDataParallel):
            dataloader.sampler.set_epoch(iepoch)
            nchunk = self.cfg.num_gpus
        peep_rate = max(10, (len(dataloader) // 10))

        ### special eval ###
        topk_bar = 8
        l_ntrue, r_ntrue, f_ntrue, b_ntrue, l_ntotal, r_ntotal, same_ntrue, same_ntotal = [0] * 8

        left_right = self.decoder_vocab(["left", "right"]) # TODO hard-coded relations
        front_behind = self.decoder_vocab(["in_front_of", "behind"])

        acc_by_attr = {k: [0, 0] for k in self.type_attrs_ext_int}
        acc_by_attr_same = {k: [0, 0] for k in self.type_attrs_ext_int}
        ### special eval ###

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

            batch_dict["infer"] = True
            batch_dict["analyze"] = True

            loss_mean, (_, loss_out) = self.model(**batch_dict)
            ntoken, loss_all = loss_out
            loss = loss_all.sum() if isinstance(loss_all, Tensor) else loss_mean * ntoken


            ### special eval ###
            (
                l_ntrue_a, r_ntrue_a, f_ntrue_a, b_ntrue_a, l_ntotal_a, r_ntotal_a,
                same_ntrue_a, same_ntotal_a, acc_by_attr_a, acc_by_attr_same_a, *_
            ) = self.evaluate_attr_and_rel(
                self.type_attrs_ext_int, left_right, front_behind, topk_bar=8, verbose=False
            )

            l_ntrue += l_ntrue_a
            r_ntrue += r_ntrue_a
            f_ntrue += f_ntrue_a
            b_ntrue += b_ntrue_a
            l_ntotal += l_ntotal_a
            r_ntotal += r_ntotal_a
            same_ntrue += same_ntrue_a
            same_ntotal += same_ntotal_a

            update_precision_from_dict(acc_by_attr, acc_by_attr_a)
            update_precision_from_dict(acc_by_attr_same, acc_by_attr_same_a)
            ### special eval ###


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
        # show special eval results
        self.show_metric(
            l_ntrue, r_ntrue, f_ntrue, b_ntrue, l_ntotal, r_ntotal,
            same_ntrue, same_ntotal, acc_by_attr, acc_by_attr_same, topk_bar=topk_bar, verbose=True
        )
        return model.report()

    def decode_batch(
        self, obj_names=None, obj_masks=None, sequences=None, obj_boxes=None,
        file_name=None, enc_extra=None, dec_extra=None, shape=(320, 480), **kwargs
    ):
        obj_count = (obj_masks == False).sum(-1).tolist()
        obj_names = [
            self.encoder_vocab(obj_names[i][:l].tolist())
            for i, l in enumerate(obj_count)
        ]
        #print(obj_names, obj_count)

        seq_count = (sequences != self.decoder_vocab.PAD_IDX).sum(-1).tolist()
        seq_names = [
            self.decoder_vocab(sequences[i][1:l].tolist()) + [self.decoder_vocab.EOS]
            for i, l in enumerate(seq_count)
        ]
        #print(seq_count, len(seq_count))

        obj_names = cat_name_list(obj_names)
        #print(obj_names)

        h, w = shape
        obj_boxes = obj_boxes.clone()
        obj_boxes[:, :, 0::2] *= w # w
        obj_boxes[:, :, 1::2] *= h # h
        obj_boxes[:, :, 2] -= obj_boxes[:, :, 0]
        obj_boxes[:, :, 3] -= obj_boxes[:, :, 1]
        obj_boxes = obj_boxes.long()
        obj_boxes = [
            obj_boxes[i][:l].tolist() for i, l in enumerate(obj_count)
        ]
        #print(obj_boxes)

        return obj_names, obj_boxes, seq_names, file_name

    def show_metric(
        self, l_ntrue, r_ntrue, f_ntrue, b_ntrue, l_ntotal, r_ntotal,
        same_ntrue, same_ntotal, acc_by_attr, acc_by_attr_same, topk_bar, verbose=False
    ):
        if not verbose:
            return
        same = same_ntrue / same_ntotal * 100
        ll = l_ntrue / l_ntotal * 100
        rr = r_ntrue / r_ntotal * 100
        ff = f_ntrue / l_ntotal * 100
        bb = b_ntrue / r_ntotal * 100

        msg = (
            f"Purity (set size {topk_bar}): {same:.2f} ({same_ntotal / topk_bar:.0f})\n" +
            f"Gold 'left' and hypothesis 'left': ({l_ntotal}) {ll:.2f}, otherwise {100 - ll:.2f}\n" +
            f"Gold 'right' and hypothesis 'right': ({r_ntotal}) {rr:.2f}, otherwise {100 - rr:.2f}\n" +
            f"Gold 'front' and hypothesis 'front': ({l_ntotal}) {ff:.2f}, otherwise {100 - ff:.2f}\n" +
            f"Gold 'behind' and hypothesis 'behind': ({r_ntotal}) {bb:.2f}, otherwise {100 - bb:.2f}"
        )

        attr_msg = estimate_precision_from_dict(acc_by_attr, " ALL")
        attr_same_msg = estimate_precision_from_dict(acc_by_attr_same, "SAME")
        self.echo(f"\n{msg}\n{attr_msg}\n{attr_same_msg}")

    def evaluate_attr_and_rel(
        self, type_attrs_ext_int, left_right, front_behind, topk_bar=8, verbose=False
    ):
        pair_logits = self.model.last_batch["dec_extra"]["pair_logit"]
        obj_names, obj_boxes, *_ = self.decode_batch(**self.model.last_batch)

        l_ntrue, r_ntrue, b_ntrue, f_ntrue, l_ntotal, r_ntotal = [0] * 6
        ntrue, ntotal, same_ntrue, same_ntotal = [0] * 4

        vocab_size = len(self.decoder_vocab)
        acc_by_attr = {k: [0, 0] for k in type_attrs_ext_int}
        acc_by_attr_same = {k: [0, 0] for k in type_attrs_ext_int}

        for ib, pair_logit in enumerate(pair_logits):
            pair_logit = pair_logit[0]

            S = int(np.sqrt(pair_logit.shape[0]))
            indice1 = torch.arange(S, device=self.device).repeat_interleave(S)
            indice2 = torch.arange(S, device=self.device).repeat(S)

            obj_list = obj_names[ib]
            box_list = obj_boxes[ib]
            assert len(obj_list) == len(box_list)

            obj_list = obj_list + [[]] * (S - len(obj_list))
            box_list = box_list + [[]] * (S - len(box_list))

            all_pairs = [obj_list[i : i + 1] + obj_list[j : j + 1] + [i, j] for i, j in zip(indice1.cpu().tolist(), indice2.cpu().tolist())]
            all_boxes = [box_list[i : i + 1] + box_list[j : j + 1] + [i, j] for i, j in zip(indice1.cpu().tolist(), indice2.cpu().tolist())]

            indicator = [True if len(pair[0]) > 0 and len(pair[1]) > 0 else False for pair in all_pairs]

            pairs = [all_pairs[iflag] for iflag, flag in enumerate(indicator) if flag]
            boxes = [all_boxes[iflag] for iflag, flag in enumerate(indicator) if flag]
            plogits = [pair_logit[iflag] for iflag, flag in enumerate(indicator) if flag]

            for pair, box, plogit in zip(pairs, boxes, plogits):
                id0, id1 = pair[2:4]
                pair = pair[:2]

                box0, box1 = box[:2]
                cx0, cy0 = box0[0] + box0[2] * 0.5, box0[1] + box0[3] * 0.5
                cx1, cy1 = box1[0] + box1[2] * 0.5, box1[1] + box1[3] * 0.5

                valid_attr = set(list(" ".join(pair).split()))
                valid_attr_id = self.decoder_vocab(list(valid_attr))

                attr_by_type = defaultdict(list)
                for attr in valid_attr:
                    attr_type = attr_types[attr]
                    attr_by_type[attr_type].append(attr)

                attr_by_type_ext = defaultdict(list)
                attr_by_type_ext_int = defaultdict(list)
                for k, v in attr_by_type.items():
                    v_new = [] + v
                    for vv in v:
                        syns = syn_attrs.get(vv, None)
                        if syns is None:
                            continue
                        v_new.extend(syns)
                    v_new = list(set(v_new))
                    attr_by_type_ext[k] = v_new
                    attr_by_type_ext_int[k] = self.decoder_vocab(v_new)

                for k, v in attr_by_type_ext_int.items():
                    assert k in type_attrs_ext_int
                    all_attr_v = type_attrs_ext_int[k]

                    a = torch.tensor([float("-inf")] * vocab_size, device=plogit.device)
                    a[all_attr_v] = plogit[all_attr_v]

                    kmax = a.argmax().cpu().item()
                    is_true = kmax in v

                    acc_by_attr[k][0] += is_true
                    acc_by_attr[k][1] += 1

                    if id0 == id1:
                        acc_by_attr_same[k][0] += is_true
                        acc_by_attr_same[k][1] += 1


                ordered, indice = plogit.sort(descending=True)
                topk = indice.cpu().tolist()[:topk_bar]

                overlap = set(valid_attr_id) & set(topk)

                ntotal += len(topk)
                ntrue += len(overlap)

                if id0 == id1:
                    same_ntrue += len(overlap)
                    same_ntotal += len(topk)

                if id0 != id1:
                    all_attr_v = left_right
                    a = torch.tensor([float("-inf")] * vocab_size, device=plogit.device)
                    a[all_attr_v] = plogit[all_attr_v]
                    kmax_lr = a.argmax().cpu().item()

                    all_attr_v = front_behind
                    a = torch.tensor([float("-inf")] * vocab_size, device=plogit.device)
                    a[all_attr_v] = plogit[all_attr_v]
                    kmax_fb = a.argmax().cpu().item()

                    if id0 < id1:
                        l_ntotal += 1
                        if kmax_lr == left_right[0]:
                            l_ntrue += 1

                        if cy0 > cy1: # gold front
                            if kmax_fb == front_behind[0]:
                                f_ntrue += 1
                        else: # gold behind
                            if kmax_fb == front_behind[1]:
                                b_ntrue += 1
                    else:
                        r_ntotal += 1
                        if kmax_lr == left_right[1]:
                            r_ntrue += 1

                        if cy0 > cy1: # gold front
                            if kmax_fb == front_behind[0]:
                                f_ntrue += 1
                        else: # gold behind
                            if kmax_fb == front_behind[1]:
                                b_ntrue += 1

        self.show_metric(
            l_ntrue, r_ntrue, f_ntrue, b_ntrue, l_ntotal, r_ntotal,
            same_ntrue, same_ntotal, acc_by_attr, acc_by_attr_same, topk_bar=topk_bar, verbose=verbose
        )

        return (
            l_ntrue, r_ntrue, f_ntrue, b_ntrue, l_ntotal, r_ntotal,
            same_ntrue, same_ntotal, acc_by_attr, acc_by_attr_same,
            pairs, torch.stack(plogits, 0)
        )
