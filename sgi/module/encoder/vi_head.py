import torch
import torch.nn as nn

from typing import Tuple
from torch import nn, Tensor

from .encoder_head import build_encoder_head
from sgi.util import Params

class BiEncInference(nn.Module):
    def __init__(
        self, cfg, token_vocab, source_token_vocab=None, target_token_vocab=None, **kwargs
    ):
        super().__init__()
        self.src_encoder = build_encoder_head(cfg.source, source_token_vocab)
        self.tgt_encoder = build_encoder_head(cfg.target, target_token_vocab)
        
        self.tgt2src = nn.Linear(
            self.tgt_encoder.output_size, self.src_encoder.output_size, bias=False
        )
        
        self.dist_type = "categorical"
        self.only_past = False # (past / present) info seen by the target encoder

    def forward(
        self, 
        src: Tensor,
        tgt: Tensor, *args, 
        bbox: Tensor=None,
        img_shape: Tuple=(320, 480),
        src_attn_mask: Tensor=None,
        src_key_padding_mask: Tensor=None,
        tgt_attn_mask: Tensor=None,
        tgt_key_padding_mask: Tensor=None,
        attn_weight_type: str=None,
        **kwargs
    ):
        assert src.dim() == 3 and tgt.dim() == 3, f"expect src/tgt being embedded as (B, L, H)"

        src_memory_bank, _, _ = self.src_encoder(
            src, *args, 
            bbox=bbox, 
            img_shape=img_shape, 
            self_attn_mask=src_attn_mask,
            self_key_padding_mask=src_key_padding_mask, 
            **kwargs 
        )

        sli = slice(None, -1) if self.only_past else slice(1, None)
        tgt = tgt[:, sli] # truncked target sequences
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask[:, sli]
        if tgt_attn_mask is not None:
            tgt_attn_mask = tgt_attn_mask[..., sli, sli]

        tgt_memory_bank, _, _ = self.tgt_encoder(
            tgt, *args, 
            self_attn_mask=tgt_attn_mask,
            self_key_padding_mask=tgt_key_padding_mask, 
            **kwargs,
        )
        tgt_memory_bank = self.tgt2src(tgt_memory_bank)

        scores = torch.bmm(
            tgt_memory_bank, src_memory_bank.transpose(1, 2)
        )

        if src_key_padding_mask is not None:
            attn_mask = src_key_padding_mask.unsqueeze(1)
            scores.masked_fill_(attn_mask, float('-inf'))

        scores = scores.transpose(0, 1)
        alpha = scores.softmax(dim=-1)
        log_alpha = scores.log_softmax(dim=-1)

        scores = Params(
            alpha=alpha, log_alpha=log_alpha, dist_type=self.dist_type,
        )
        return scores
