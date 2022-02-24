from omegaconf import OmegaConf
import os, re
import torch
from torch import nn

import torch.distributed as dist
from torch.nn.parallel import data_parallel

from ..util import Stats, Statistics, enable_print, disable_print
from ..module import build_encoder_head, build_decoder_head, build_loss_head
from ..module import RNNDecoderBase
from . import load_checkpoint

class ViSGI(nn.Module):
    def __init__(self, cfg, echo):
        super().__init__()
        self.cfg = cfg
        self.echo = echo
        self.meter_train = Statistics()
        self.meter_infer = Statistics()

    def forward(
        self, 
        obj_idxes, obj_boxes, obj_names, obj_masks, sequences, img_shape,
        *args, file_name=None, obj_names_src=None, analyze=False, **kwargs
    ):
        device_ids = kwargs.get("device_ids", [0])
        # not necessary
        relations = self.relation_head(
            obj_names, *img_shape, bbox=obj_boxes, self_key_padding_mask=obj_masks, **kwargs
        )
        if isinstance(relations, dict):
            kwargs.update(relations)

        # infer q(a | x, y) 
        seq_embed = self.decoder_head.token_embed(sequences)
        seq_masks = sequences == self.decoder_head.token_vocab.PAD_IDX
        q_scores = self.vi_head(
            obj_names, 
            seq_embed, 
            bbox=obj_boxes, 
            src_key_padding_mask=obj_masks,
            tgt_key_padding_mask=seq_masks,
            **kwargs,
        ) if self.decoder_head.mode not in {"exact", "none"} else None # q

        # encode source
        objects, rels, enc_extra = self.encoder_head(
            obj_names, *img_shape, bbox=obj_boxes, self_key_padding_mask=obj_masks, **kwargs
        ) # img_shape is absorted into *args 

        # make initial states of the decoder
        encoder_final = rels # `rels' is `encoder_final' when `encoder_head' is an RNN
        # do nothing before initialization and detach every batch thereafter
        #self.decoder_head.detach_state() # no need to detach as we re-initialize every epoch
        self.decoder_head.init_state_from_memo(objects, obj_masks) # initialize the input feed

        # infer p(a | x, c)
        # `dec_outs' are not necessarilly the logits; they could be the input to softmax
        # dec_outs, target, {"dist_info": dist_info, "baselines": baselines, "attns": attns)
        dec_outs, targets, dec_extra = self.decoder_head(
            sequences, memory=objects, memo_key_padding_mask=obj_masks, q_scores=q_scores, **kwargs
        ) # p

        # TODO hack: need the predictor of decoder to project `dec_outs'
        dec_extra["decoder"] = self.decoder_head
        loss, outs = self.loss_head(dec_outs, targets, **dec_extra)
        stats = outs[-1]

        # peek the progress
        meter = self.meter_train if self.training else self.meter_infer
        meter.update(stats)

        if analyze:
            self.analyze(
                obj_names=obj_names_src,
                obj_masks=obj_masks, sequences=sequences,
                obj_boxes=obj_boxes, file_name=file_name, 
                enc_extra=enc_extra, dec_extra=dec_extra,
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
            self.vi_head.state_dict(),
        )

    def stats(self): 
        meter = self.meter_train if self.training else self.meter_infer
        return meter.report()

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
            (
                local_cfg, backbone_head_sd, relation_head_sd, 
                encoder_head_sd, decoder_head_sd, loss_head_sd, vi_head_sd,
            ) = load_checkpoint(self.cfg, self.echo)

            self.relation_head = build_encoder_head(local_cfg.model.relation, encoder_vocab)
            self.relation_head.load_state_dict(relation_head_sd)

            self.encoder_head = build_encoder_head(local_cfg.model.encoder, encoder_vocab)
            self.encoder_head.load_state_dict(encoder_head_sd)

            self.decoder_head = build_decoder_head(local_cfg.model.decoder, decoder_vocab)
            self.decoder_head.load_state_dict(decoder_head_sd)

            self.loss_head = build_loss_head(local_cfg.model.loss, decoder_vocab)
            self.loss_head.load_state_dict(loss_head_sd)

            self.vi_head = build_encoder_head(local_cfg.model.vi, decoder_vocab)
            self.vi_head.load_state_dict(vi_head_sd)
        else:
            self.relation_head = build_encoder_head(self.cfg.model.relation, encoder_vocab)
            self.encoder_head = build_encoder_head(self.cfg.model.encoder, encoder_vocab)
            self.decoder_head = build_decoder_head(self.cfg.model.decoder, decoder_vocab)
            self.vi_head = build_encoder_head(
                self.cfg.model.vi, None, source_token_vocab=encoder_vocab, target_token_vocab=decoder_vocab,
            )
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
                f"vi_head.{k}": v for k, v in self.vi_head.named_parameters()
            })
            tunable_params.update({
                f"loss_head.{k}": v for k, v in self.loss_head.named_parameters()
            })
        self.cuda(self.cfg.rank)
        return tunable_params

