import torch
import torch.nn as nn

from typing import Tuple
from torch import nn, Tensor
from collections import defaultdict

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from sgi.util import flip_and_shift, bidirectional_causality
from .. import build_attention_head
from .rnn_base import BiRNNDecoderBase


class CatThenAttendRNNDecoderHead(BiRNNDecoderBase):
    """ we concatenate left and right contexts of a word to select source contexts  
    """
    def __init__(self, cfg, token_vocab, tgt_dim=None):
        super().__init__(cfg, token_vocab, bidirectional=True)
        assert self.batch_first, f"expect `batch_first' inputs"
        self.encoder = self._build_rnn(cfg.rnn_type, **self.rnn_params)

        assert cfg.attention is not None, f"attention configurations are missing"
        self.attn = build_attention_head(
            cfg.attention, tgt_dim=(tgt_dim or self.hidden_size * 2)
        )

        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_size * 2, len(self.token_vocab))
        )

    @property
    def _input_size(self):
        return self.emb_size

    def init_state_from_memo(self, memo, mask):
        B, _, H = memo.shape
        device = memo.device
        if isinstance(self.encoder, nn.LSTM):
            h = torch.zeros(2 * self.num_layers, B, H, device=device)
            c = torch.zeros(2 * self.num_layers, B, H, device=device)
            hidden = (h, c)
        else: # (D * num_layers, B, H)
            h = torch.zeros(2 * self.num_layers, B, H, device=device)
            hidden = (h,)
        self.state["hidden"] = hidden

    def _run_forward_pass(
        self, x, lengths, memory, memory_mask=None, enforce_sorted=False
    ):
        assert False, f"there is bug yet to be fixed."
        # FIXME when num_layers > 1 the encoder will break causality.
        """
        length_list = lengths.cpu().view(-1).tolist()
        packed_emb = pack_padded_sequence(
            x, length_list, batch_first=self.batch_first, enforce_sorted=enforce_sorted
        )
        dec_state = self.state["hidden"]
        rnn_outs, _ = self.encoder(packed_emb, dec_state)
        rnn_outs, _ = pad_packed_sequence(rnn_outs, batch_first=self.batch_first)
        
        # CAT create causal repr.
        forward, reverse = rnn_outs.chunk(2, dim=-1)
        rnn_outs = bidirectional_causality(forward, reverse)
        """

        # ATTEND source context selection
        attns = defaultdict(list)
        dec_outs = []

        step_dim = 1 # memory-efficient
        for i, rnn_output in enumerate(rnn_outs.split(1, dim=step_dim)):
            rnn_output = rnn_output.squeeze(step_dim)
            decoder_output, p_attn, _ = self.attn(
                rnn_output, memory, memory_mask=memory_mask
            )
            dec_outs.append(self.dropout(decoder_output))
            attns["std"].append(p_attn)

        # concat sequence of tensors along the time dim 
        dec_outs = torch.stack(dec_outs, dim=1) # (B, L, H)
        for k in attns:
            attns[k] = torch.stack(attns[k])
        return dec_outs, attns


class AttendThenCatRNNDecoderHead(CatThenAttendRNNDecoderHead):
    """ we concatenate left and right contexts of a word to select source contexts  
    """
    def __init__(self, cfg, token_vocab):
        super().__init__(cfg, token_vocab, tgt_dim=cfg.hidden_size)

    def _run_forward_pass(
        self, x, lengths, memory, memory_mask=None, enforce_sorted=False
    ):
        length_list = lengths.cpu().view(-1).tolist()
        packed_emb = pack_padded_sequence(
            x, length_list, batch_first=self.batch_first, enforce_sorted=enforce_sorted
        )
        dec_state = self.state["hidden"]
        rnn_outs, _ = self.encoder(packed_emb, dec_state)
        rnn_outs, _ = pad_packed_sequence(rnn_outs, batch_first=self.batch_first)
        
        forward, reverse = rnn_outs.chunk(2, dim=-1)
        attns = defaultdict(list)
        forward_dec_outs = []
        reverse_dec_outs = []

        def _encode(x, dec_outs, attns, attn_key, step_dim=1):
            for i in range(x.shape[step_dim]):
                rnn_output = x[:, i]
                decoder_output, p_attn, _ = self.attn(
                    rnn_output, memory, memory_mask=memory_mask
                )
                dec_outs.append(self.dropout(decoder_output))
                attns[attn_key].append(p_attn)

        # ATTEND source context selection
        _encode(forward, forward_dec_outs, attns, "std_forward")
        _encode(reverse, reverse_dec_outs, attns, "std_reverse")

        # CAT create causal repr.
        forward = torch.stack(forward_dec_outs, dim=1) # (B, L, H)
        reverse = torch.stack(reverse_dec_outs, dim=1) # (B, L, H)
        reverse = flip_and_shift(reverse, lengths)

        dec_outs = bidirectional_causality(forward, reverse)

        for k in attns:
            attns[k] = torch.stack(attns[k])
        return dec_outs, attns


class BiInputFeedRNNDecoderHead(BiRNNDecoderBase):
    """ we concatenate left and right contexts of a word to select source contexts  
    """
    def __init__(self, cfg, token_vocab):
        super().__init__(cfg, token_vocab, bidirectional=False)
        assert self.batch_first, f"expect `batch_first' inputs"
        self.forward_encoder = self._build_rnn(cfg.rnn_type, **self.rnn_params)
        self.reverse_encoder = self._build_rnn(cfg.rnn_type, **self.rnn_params)

        assert cfg.attention is not None, f"attention configurations are missing"
        self.attn = build_attention_head(cfg.attention, tgt_dim=self.hidden_size)

        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_size * 2, len(self.token_vocab))
        )

    @property
    def _input_size(self):
        return self.emb_size + self.hidden_size

    def init_state_from_memo(self, memo, mask):
        B, L, H = memo.shape
        device = memo.device
        if isinstance(self.forward_encoder, nn.LSTM):
            h = torch.zeros(self.num_layers, B, H, device=device)
            c = torch.zeros(self.num_layers, B, H, device=device)
            hidden = (h, c)
        else: # (D * num_layers, B, H)
            h = torch.zeros(self.num_layers, B, H, device=device)
            hidden = (h,)
        self.state["hidden"] = hidden
        self.state["input_feed"] = self._average_memo(memo, mask)

    def _run_forward_pass(
        self, x, lengths, memory, memory_mask=None, enforce_sorted=False
    ):
        B, L, H = x.shape
        x_reverse = flip_and_shift(x, lengths)

        attns = defaultdict(list)
        forward_dec_outs = []
        reverse_dec_outs = []

        dec_state = self.state["hidden"]
        input_feed = self.state["input_feed"]

        def _encode(x, encoder, dec_state, input_feed, dec_outs, attns, attn_key, step_dim=1):
            for i in range(x.shape[step_dim]):
                emb = x[:, i]
                decoder_input = torch.cat([emb, input_feed], 1).unsqueeze(step_dim) 

                rnn_output, dec_state = encoder(decoder_input, dec_state)
                rnn_output = rnn_output.squeeze(step_dim)

                decoder_output, p_attn, input_feed = self.attn(
                    rnn_output, memory, memory_mask=memory_mask
                )

                dec_outs.append(self.dropout(decoder_output))
                attns[attn_key].append(p_attn)

        _encode(x, self.forward_encoder, dec_state, input_feed, forward_dec_outs, attns, "std_forward")
        _encode(x_reverse, self.reverse_encoder, dec_state, input_feed, reverse_dec_outs, attns, "std_reverse")

        # create causal repr.
        forward = torch.stack(forward_dec_outs, dim=1) # (B, L, H)
        reverse = torch.stack(reverse_dec_outs, dim=1) # (B, L, H)
        reverse = flip_and_shift(reverse, lengths)

        dec_outs = bidirectional_causality(forward, reverse)

        for k in attns:
            attns[k] = torch.stack(attns[k])
        return dec_outs, attns

