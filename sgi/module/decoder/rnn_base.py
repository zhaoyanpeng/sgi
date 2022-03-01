import torch
import torch.nn as nn

from typing import Tuple
from torch import nn, Tensor

from .. import rnn_factory
from .decoder_head import MetaDecHead

class RNNDecoderBase(MetaDecHead):
    def __init__(self, cfg, token_vocab, bidirectional=False):
        super(RNNDecoderBase, self).__init__(cfg, token_vocab)
        self.hidden_size = cfg.hidden_size
        self.rnn_params = {
            "bidirectional": bidirectional,
            "batch_first": cfg.batch_first,
            "hidden_size": cfg.hidden_size,
            "input_size": self._input_size,
            "num_layers": cfg.num_layers,
            "dropout": cfg.dropout,
        }
        self.dropout = nn.Dropout(cfg.dropout)
        self.encoder = None
        # decoder state
        self.state = {}
        # compatibility
        self.mode = "none"
        self.attn = None 

    @property
    def bidirectional(self):
        return self.rnn_params["bidirectional"]

    @property
    def batch_first(self):
        return self.rnn_params["batch_first"]

    @property
    def num_layers(self):
        return self.rnn_params["num_layers"]

    def _average_memo(self, memo, mask):
        # average the source contexts
        B, L, H = memo.shape
        lengths = (
            torch.tensor([1.] * B, device=memo.device)
            if mask is None else (~mask).sum(-1)
        )
        memo = memo.sum(1) / lengths.unsqueeze(-1) # (B, H)
        return memo

    def init_state_from_memo(self, memo, mask):
        return NotImplementedError

    def _build_rnn(self, rnn_type, **kwargs):
        rnn, _ = rnn_factory(rnn_type, **kwargs)
        return rnn

    def update_dropout(self, dropout):
        self.dropout.p = dropout
        if self.encoder is not None:
            self.encoder.dropout.p = dropout

    def init_state_from_memo(self, memo, mask):
        # do nothing if the state has been initialized at the end of `forward'
        # why do we preferably use the last-batch statistics?
        if len(self.state) > 0:
            pass #return
        B, L, H = memo.shape
        device = memo.device
        # decoder must has a single direction so we only need the layer number
        if isinstance(self.encoder, nn.LSTM):
            h = torch.zeros(self.num_layers, B, H, device=memo.device)
            #torch.stack([memo] * self.num_layers, dim=0)
            c = torch.zeros(self.num_layers, B, H, device=memo.device)
            #torch.stack([memo] * self.num_layers, dim=0)
            hidden = (h, c)
        else: # (D * num_layers, B, H)
            h = torch.zeros(self.num_layers, B, H, device=memo.device)
            #torch.stack([memo] * self.num_layers, dim=0)
            hidden = (h,)
        self.state["hidden"] = hidden
        self.state["input_feed"] = self._average_memo(memo, mask)

    def init_state(self, encoder_final):
        """Initialize decoder state with last state of the encoder."""
        if len(self.state) > 0:
            return # has been initialized
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional:
                hidden = torch.cat(
                    [hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]]
                , 2)
            return hidden

        if isinstance(encoder_final, tuple):  # LSTM
            self.state["hidden"] = tuple(
                _fix_enc_hidden(enc_hid) for enc_hid in encoder_final
            )
        else:  # GRU
            self.state["hidden"] = (_fix_enc_hidden(encoder_final), )

        # Init the input feed.
        batch_size = self.state["hidden"][0].size(1)
        h_size = (batch_size, self.hidden_size)
        self.state["input_feed"] = \
            self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)

    def detach_state(self):
        if "hidden" in self.state:
            self.state["hidden"] = tuple(h.detach() for h in self.state["hidden"])
        if "input_feed" in self.state:
            self.state["input_feed"] = self.state["input_feed"].detach()

    def forward( 
        self, 
        x: Tensor,
        memory: Tensor=None,
        memo_attn_mask: Tensor=None,
        memo_key_padding_mask: Tensor=None,
        enforce_sorted: bool=False,
        **kwargs,
    ):
        assert x.dim() == 2, f"expect x of shape (B, L)"
        input_x = x[:, :-1] # causal LM sees the past
        x_embed = self._encode_positions(input_x)

        x_mask = x == self.token_vocab.PAD_IDX
        x_lengths = (~x_mask).sum(-1) - 1

        dec_state, dec_outs, attns = self._run_forward_pass(
            x_embed, x_lengths, memory,
            memory_mask=memo_key_padding_mask, enforce_sorted=enforce_sorted,
        )

        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        self.state["hidden"] = dec_state
        
        # batch-second outputs
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs)
        for k in attns:
            if type(attns[k]) == list:
                attns[k] = torch.stack(attns[k])

        gold_x = x[:, 1:].transpose(0, 1).contiguous()
        #dec_outs = self.predictor(dec_outs)
        loss_state = {"output": dec_outs, "target": gold_x}
        return dec_outs, gold_x, {"loss_state": loss_state, "attns": attns}


class BiRNNDecoderBase(MetaDecHead):
    def __init__(self, cfg, token_vocab, bidirectional=False):
        super(BiRNNDecoderBase, self).__init__(cfg, token_vocab)
        self.hidden_size = cfg.hidden_size
        self.rnn_params = {
            "bidirectional": bidirectional,
            "batch_first": cfg.batch_first,
            "hidden_size": cfg.hidden_size,
            "input_size": self._input_size,
            "num_layers": cfg.num_layers,
            "dropout": cfg.dropout,
        }
        self.dropout = nn.Dropout(cfg.dropout)
        # decoder state
        self.state = {}
        # compatibility
        self.mode = "none"
        self.attn = None 

    @property
    def bidirectional(self):
        return self.rnn_params["bidirectional"]

    @property
    def batch_first(self):
        return self.rnn_params["batch_first"]

    @property
    def num_layers(self):
        return self.rnn_params["num_layers"]

    def _average_memo(self, memo, mask):
        # average the source contexts
        B, L, H = memo.shape
        lengths = (
            torch.tensor([1.] * B, device=memo.device)
            if mask is None else (~mask).sum(-1)
        )
        memo = memo.sum(1) / lengths.unsqueeze(-1) # (B, H)
        return memo

    def init_state_from_memo(self, memo, mask):
        return NotImplementedError

    def _build_rnn(self, rnn_type, **kwargs):
        rnn, _ = rnn_factory(rnn_type, **kwargs)
        return rnn

    def forward(
        self,
        x: Tensor,
        memory: Tensor=None,
        memo_attn_mask: Tensor=None,
        memo_key_padding_mask: Tensor=None,
        enforce_sorted: bool=False,
        **kwargs,
    ):
        x_embed = self._encode_positions(x)

        x_mask = x == self.token_vocab.PAD_IDX
        x_lengths = (~x_mask).sum(-1)

        dec_outs, attns = self._run_forward_pass(
            x_embed, x_lengths, memory,
            memory_mask=memo_key_padding_mask, enforce_sorted=enforce_sorted,
        )

        gold_x = x #[:, 1:-1]
        dec_outs = self.predictor(dec_outs)
        loss_state = {"output": dec_outs, "target": gold_x}

        return dec_outs, gold_x, {"loss_state": loss_state, "attns": attns}
