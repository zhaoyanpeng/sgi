import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from .rnn_base import RNNEncoderBase

class RNNEncoder(RNNEncoderBase):
    def __init__(self, cfg, token_vocab, **kwargs):
        super().__init__(cfg, token_vocab, bidirectional=cfg.bidirectional)
        self.encoder = self._build_rnn(cfg.rnn_type, **self.rnn_params)

    def forward(
        self, 
        x: Tensor, *args,
        bbox: Tensor=None,
        img_shape: Tuple=(320, 480),
        self_attn_mask: Tensor=None,
        self_key_padding_mask: Tensor=None,
        attn_weight_type: str=None,
        enforce_sorted: bool=False,
        **kwargs
    ):
        # x is expected to have been encoded by an external backbone
        # and `_encode_position' will ignore all the embedding stuff
        # batch, s_len, emb_dim = x.size()
        # assert x.dim() == 3, f"Please embed `x' first."
        x_embed = self._encode_positions(x, bbox, img_shape)

        lengths = (
            None if self_key_padding_mask is None 
            else (~self_key_padding_mask).sum(-1) 
        )

        packed_emb = x_embed
        if lengths is not None:
            length_list = lengths.cpu().view(-1).tolist()
            packed_emb = pack_padded_sequence(
                x_embed, length_list, batch_first=self.batch_first, enforce_sorted=enforce_sorted
            )

        rnn_outs, rnn_final = self.encoder(packed_emb)

        if lengths is not None:
            rnn_outs, _ = pad_packed_sequence(rnn_outs, batch_first=self.batch_first)

        return rnn_outs, rnn_final, {}
