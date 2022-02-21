from omegaconf import OmegaConf
import os, re
import torch
from torch import nn

import torch.distributed as dist
from torch.nn.parallel import data_parallel

from ..util import Stats, enable_print, disable_print
from ..module import build_encoder_head, build_decoder_head, build_loss_head
from . import load_checkpoint

class SGI(nn.Module):
    def __init__(self, cfg, echo):
        super().__init__()
        self.cfg = cfg
        self.echo = echo
        self.meter_train = Stats()
        self.meter_infer = Stats()

    def forward(
        self, 
        obj_idxes, obj_boxes, obj_names, obj_masks, sequences, img_shape,
        *args, file_name=None, obj_names_int=None, analyze=False, **kwargs
    ):
        device_ids = kwargs.get("device_ids", [0])
        relations = self.relation_head(
            obj_names, *img_shape, bbox=obj_boxes, self_key_padding_mask=obj_masks, **kwargs
        )
        if isinstance(relations, dict):
            kwargs.update(relations)
        objects, rels, enc_extra = self.encoder_head(
            obj_names, *img_shape, bbox=obj_boxes, self_key_padding_mask=obj_masks, **kwargs
        ) 
        logits, targets, dec_extra = self.decoder_head(
            sequences, memory=objects, memo_key_padding_mask=obj_masks
        )
        loss, outs = self.loss_head(logits, targets)
        if analyze:
            self.analyze(
                obj_names=obj_names_int, obj_masks=obj_masks, sequences=sequences,
                obj_boxes=obj_boxes, file_name=file_name, enc_extra=enc_extra, dec_extra=dec_extra,
            )
        return loss, (rels, outs) 

    def analyze(self, **kwargs):
        self.last_batch = {k: v for k, v in kwargs.items()}
        return None

    def collect_state_dict(self):
        return (
            self.backbone_head.state_dict() if self.backbone_head is not None else {}, 
            self.relation_head.state_dict(), 
            self.encoder_head.state_dict(), 
            self.decoder_head.state_dict(), 
            self.loss_head.state_dict(),
        )

    def stats(self): 
        return ""

    def reset(self):
        meter = self.meter_train if self.training else self.meter_infer
        meter.reset()
    
    def reduce_grad(optim_rate, sync=False):
        raise NotImplementedError("Gradient Reduce")

    def report(self, gold_file=None):
        if (not dist.is_initialized() or dist.get_rank() == 0) and self.loss_head is not None:
            return self.loss_head.report(gold_file=gold_file) 
        else:
            return ""
    
    def backbone(self, obj_names, obj_name_lengths):
        obj_names = self.encoder_head.token_embed(obj_names)
        if not self.cfg.model.encoder.cat_w:
            obj_names = obj_names.sum(-2) / obj_name_lengths.unsqueeze(-1) 
        else: # concatenate attribute words
            shape = obj_names.size()[:2] + (-1,)
            obj_names = obj_names.view(shape)
        return obj_names
    
    def build(self, encoder_vocab, decoder_vocab):
        tunable_params = dict()
        self.backbone_head = None
        if self.cfg.eval:
            local_cfg, backbone_head_sd, relation_head_sd, encoder_head_sd, decoder_head_sd, loss_head_sd = \
                load_checkpoint(self.cfg, self.echo)

            self.relation_head = build_encoder_head(local_cfg.model.relation, encoder_vocab)
            self.relation_head.load_state_dict(relation_head_sd)

            self.encoder_head = build_encoder_head(local_cfg.model.encoder, encoder_vocab)
            self.encoder_head.load_state_dict(encoder_head_sd)

            self.decoder_head = build_decoder_head(local_cfg.model.decoder, decoder_vocab)
            self.decoder_head.load_state_dict(decoder_head_sd)

            self.loss_head = build_loss_head(self.cfg.model.loss, decoder_vocab)
            self.loss_head.load_state_dict(loss_head_sd)
        else:
            self.relation_head = build_encoder_head(self.cfg.model.relation, encoder_vocab)
            self.encoder_head = build_encoder_head(self.cfg.model.encoder, encoder_vocab)
            self.decoder_head = build_decoder_head(self.cfg.model.decoder, decoder_vocab)
            self.loss_head = build_loss_head(self.cfg.model.loss, decoder_vocab)
            tunable_params = {
                f"relation_head.{k}": v for k, v in self.relation_head.named_parameters()
            } 
            tunable_params.update({
                f"encoder_head.{k}": v for k, v in self.encoder_head.named_parameters()
            })
            tunable_params.update({
                f"decoder_head.{k}": v for k, v in self.decoder_head.named_parameters()
            })
            tunable_params.update({
                f"loss_head.{k}": v for k, v in self.loss_head.named_parameters()
            })
        self.cuda(self.cfg.rank)
        return tunable_params

