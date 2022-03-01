import torch
import torch.nn as nn

from typing import Tuple
from torch import nn, Tensor
from collections import defaultdict

from .. import build_attention_head
from .rnn_base import RNNDecoderBase


class StdRNNDecoder(RNNDecoderBase):
    def __init__(self, cfg, token_vocab, tgt_dim=None):
        super().__init__(cfg, token_vocab, bidirectional=False)
        assert self.batch_first, f"expect `batch_first' inputs"
        self.encoder = self._build_rnn(cfg.rnn_type, **self.rnn_params)

        if cfg.attention is not None:
            self.attn = build_attention_head(
                cfg.attention, tgt_dim=(tgt_dim or self.hidden_size)
            )

        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_size, len(self.token_vocab))
        )

    @property
    def _input_size(self):
        return self.emb_size

    def _run_forward_pass(
        self, x, lengths, memory, memory_mask=None, enforce_sorted=False
    ):
        length_list = lengths.cpu().view(-1).tolist()
        packed_emb = pack_padded_sequence(
            x, length_list, batch_first=self.batch_first, enforce_sorted=enforce_sorted
        )
        dec_state = self.state["hidden"]
        dec_state = dec_state[0] if isinstance(self.encoder, nn.GRU) else dec_state
        rnn_outs, dec_state = self.encoder(packed_emb, dec_state)
        rnn_output, _ = pad_packed_sequence(rnn_outs, batch_first=self.batch_first)

        attns = {}
        dec_outs = rnn_output
        if self.attn is not None:
            dec_outs, p_attn, _ = self.attn(
                rnn_output, memory, memory_mask=memory_mask
            )
            attns["std"] = p_attn
        dec_outs = self.dropout(dec_outs)
        return dec_state, dec_outs, attns


class InputFeedRNNDecoder(RNNDecoderBase):
    """Input feeding based decoder.
    """
    def __init__(self, cfg, token_vocab, tgt_dim=None):
        super().__init__(cfg, token_vocab, bidirectional=False)
        assert self.batch_first, f"expect `batch_first' inputs"
        self.encoder = self._build_rnn(cfg.rnn_type, **self.rnn_params)

        assert cfg.attention is not None, f"attention configurations are missing"
        self.attn = build_attention_head(
            cfg.attention, tgt_dim=(tgt_dim or self.hidden_size)
        )

        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_size, len(self.token_vocab))
        )

    @property
    def _input_size(self):
        return self.emb_size + self.hidden_size

    def _run_forward_pass(
        self, x, lengths, memory, memory_mask=None, enforce_sorted=False
    ):
        dec_state = self.state["hidden"]
        dec_state = dec_state[0] if isinstance(self.encoder, nn.GRU) else dec_state
        input_feed = self.state["input_feed"]

        step_dim = 1 # as we have batch-first input
        attns = defaultdict(list)
        dec_outs = []

        for i, emb in enumerate(x.split(1, dim=step_dim)):
            emb = emb.squeeze(step_dim)
            decoder_input = torch.cat([emb, input_feed], 1).unsqueeze(step_dim)
            rnn_output, dec_state = self.encoder(decoder_input, dec_state)
            rnn_output = rnn_output.squeeze(step_dim)

            decoder_output, p_attn, input_feed = self.attn(
                rnn_output, memory, memory_mask=memory_mask
            )
            dec_outs.append(self.dropout(decoder_output))
            attns["std"].append(p_attn)
        return dec_state, dec_outs, {"attns": attns}
