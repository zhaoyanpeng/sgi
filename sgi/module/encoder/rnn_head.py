import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from .. import rnn_factory
from .encoder_head import MetaEncHead

class RNNEncoder(MetaEncHead):
    """ A generic recurrent neural network encoder.
    """
    def __init__(self, cfg, token_vocab, **kwargs):
        super().__init__(cfg, token_vocab)
        self.hidden_size = cfg.hidden_size
        num_directions = 2 if cfg.bidirectional else 1
        assert cfg.hidden_size % num_directions == 0
        hidden_size = cfg.hidden_size // num_directions

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(
                cfg.rnn_type,
                input_size=self._input_size,
                hidden_size=hidden_size,
                num_layers=cfg.num_layers,
                bidirectional=cfg.bidirectional,
                dropout=cfg.dropout,
            )

    def forward(
        self, 
        x: Tensor, *args,
        bbox: Tensor=None,
        img_shape: Tuple=(320, 480),
        self_attn_mask: Tensor=None,
        self_key_padding_mask: Tensor=None,
        attn_weight_type: str=None,
        enforce_sorted=False,
        batch_first=True, 
        **kwargs
    ):
        # x is expected to have been encoded by an external backbone
        # and `_encode_position' will ignore all embedding stuff
        # batch, s_len, emb_dim = x.size()
        x = self._encode_positions(x, bbox, img_shape)

        lengths = (
            None if self_key_padding_mask is None 
            else (~self_key_padding_mask).sum(-1) 
        )

        packed_emb = emb = x 
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths_list = lengths.cpu().view(-1).tolist()
            packed_emb = pack_padded_sequence(
                emb, lengths_list, batch_first=batch_first, enforce_sorted=enforce_sorted
            )

        memory_bank, encoder_final = self.rnn(packed_emb)

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank, _ = pad_packed_sequence(memory_bank, batch_first=batch_first)

        return memory_bank, encoder_final, lengths

    @property
    def _input_size(self):
        return self.emb_size

    @property
    def output_size(self):
        return self.hidden_size

    def update_dropout(self, dropout):
        self.rnn.dropout = dropout
