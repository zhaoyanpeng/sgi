from omegaconf import OmegaConf
import math
import os, re
import torch
from torch import nn

import torch.distributed as dist
from torch.nn.parallel import data_parallel

from ..module import MiniTF, MiniTFBlock, MiniTFAttention
from ..module import SelfattentionMask, PositionalEncoder

class MiniTFLM(nn.Module):
    def __init__(self, cfg, echo):
        super().__init__()
        self.cfg = cfg
        self.echo = echo

    def forward(self, src, tgt):
        """ Batch-first src: (B, L).
        """
        length = src.shape[1]
        src = self.encoder(src) * math.sqrt(self.cfg.model.encoder.m_dim)
        src = self.ln0(self.pos_encoder(src))
        self_attn_mask = (torch.triu(
            torch.ones(length, length, dtype=torch.uint8, device=src.device), 
        diagonal=1) == 1)
        output, _ = self.transformer(src, self_attn_mask=self_attn_mask)
        output = self.decoder(output)
        loss = self.loss_fn(output.view(-1, output.shape[-1]), tgt)
        return loss, None 

    def collect_state_dict(self):
        return (
            self.state_dict(), 
        )

    def stats(self): 
        return ""

    def reset(self):
        pass
    
    def reduce_grad(optim_rate, sync=False):
        raise NotImplementedError("Gradient Reduce")

    def report(self, gold_file=None):
        return ""

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def build(self, encoder_vocab, decoder_vocab):
        tunable_params = dict()
        if self.cfg.eval:
            pass
        else:
            cfg = self.cfg.model.encoder
            self.model_type = 'Transformer'
            self.pos_encoder = PositionalEncoder(cfg.m_dim, cfg.p_dropout)
            self.ln0 = nn.LayerNorm(cfg.m_dim)
            layer_fn = lambda : MiniTFBlock(
                cfg.m_dim, cfg.num_head, cfg.f_dim, cfg.attn_cls_intra, 
                attn_cls_inter=cfg.attn_cls_inter, 
                dropout=cfg.t_dropout, 
                qk_scale=cfg.qk_scale,
                activation=cfg.activation,
                attn_dropout=cfg.attn_dropout,
                proj_dropout=cfg.proj_dropout,
                num_head_intra=cfg.num_head_intra,
                num_head_inter=cfg.num_head_inter,
            )
            self.transformer = MiniTF(layer_fn, cfg.num_layer) 

            ntoken = len(encoder_vocab)
            self.decoder = nn.Linear(cfg.m_dim, ntoken)
            self.encoder = nn.Embedding(ntoken, cfg.w_dim)
            self.loss_fn = nn.CrossEntropyLoss()

            tunable_params = {
                f"{k}": v for k, v in self.named_parameters()
            } 
        self.cuda(self.cfg.rank)
        return tunable_params
