import torch
import torch.nn as nn

from typing import Tuple
from torch import nn, Tensor

from sgi.util import aeq
from .. import rnn_factory, GlobalAttention
from .decoder_head import MetaDecHead

class DecoderBase(MetaDecHead):
    """Abstract class for decoders.
    Args:
        attentional (bool): The decoder returns non-empty attention.
    """

    def __init__(self, cfg, token_vocab, attentional=True):
        super(DecoderBase, self).__init__(cfg, token_vocab)
        self.attentional = attentional

class RNNDecoderBase(DecoderBase):

    def __init__(self, cfg, token_vocab, skip_attn=False):
        super(RNNDecoderBase, self).__init__(
            cfg, token_vocab, attentional=cfg.attention is not None
        )

        self.bidirectional = False
        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_layers
        self.dropout = nn.Dropout(cfg.dropout)

        # Decoder state
        self.state = {}

        # Build the RNN.
        self.rnn = self._build_rnn(
            cfg.rnn_type,
            input_size=self._input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout
        )

        # Set up the standard attention.
        if not self.attentional or skip_attn:
            self.attn = None
        else:
            self.attn = GlobalAttention(
                cfg.hidden_size, attn_type=cfg.attn_type, attn_func=cfg.attn_func
            )

        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_size, len(self.token_vocab))
        )

    def init_state_from_memo(self, memo, mask):
        # do nothing if the state has been initialized at the end of `forward'
        # is there a better way to initialize states, why preferably use last-batch statistics?
        if len(self.state) > 0:
            pass #return

        # the source contexts
        B, L, H = memo.shape
        lengths = (
            torch.tensor([1.] * B, device=memo.device)
            if mask is None else (~mask).sum(-1)
        )
        memo = memo.sum(1) / lengths.unsqueeze(-1) # (B, H)

        # decoder must has a single direction so we only need the layer number
        if isinstance(self.rnn, nn.LSTM):
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

        # init the input feed (source contexts)
        batch_size = self.state["hidden"][0].size(1)
        h_size = (batch_size, self.hidden_size)
        self.state["input_feed"] = memo.unsqueeze(0)

    def init_state(self, encoder_final):
        """Initialize decoder state with last state of the encoder."""
        if len(self.state) > 0:
            return # has been initialized
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)
            return hidden

        if isinstance(encoder_final, tuple):  # LSTM
            self.state["hidden"] = tuple(_fix_enc_hidden(enc_hid)
                                         for enc_hid in encoder_final)
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
        tgt: Tensor,
        memory: Tensor=None,
        memo_attn_mask: Tensor=None,
        memo_key_padding_mask: Tensor=None,
        **kwargs,
    ):
        """
        Args:
            tgt (LongTensor): sequences of padded tokens
                 ``(tgt_len, batch, nfeats)``.
            memory (FloatTensor): vectors from the encoder
                 ``(src_len, batch, hidden)``.
            memory_lengths (LongTensor): the padded source lengths
                ``(batch,)``.
        Returns:
            (FloatTensor, dict[str, FloatTensor]):
            * dec_outs: output from the decoder (after attn)
              ``(tgt_len, batch, hidden)``.
            * attns: distribution over src at each tgt
              ``(tgt_len, batch, src_len)``.
        """
        memory_lengths = (
            None if memo_key_padding_mask is None 
            else (~memo_key_padding_mask).sum(-1) 
        )
        dec_state, dec_outs, attns = self._run_forward_pass(
            tgt, memory, memory_lengths=memory_lengths)

        # Update the state with the result.
        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        self.state["hidden"] = dec_state
        self.state["input_feed"] = dec_outs[-1].unsqueeze(0)

        # Concatenates sequence of tensors along a new dimension.
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs)

            for k in attns:
                if type(attns[k]) == list:
                    attns[k] = torch.stack(attns[k])
        return dec_state, dec_outs, {"attns": attns}

    def update_dropout(self, dropout):
        self.dropout.p = dropout


class StdRNNDecoder(RNNDecoderBase):
    """Standard fully batched RNN decoder with attention.
    """
    def _run_forward_pass(self, tgt, memory_bank, memory_lengths=None):
        attns = {}
        emb = self._encode_positions(tgt)

        if isinstance(self.rnn, nn.GRU):
            rnn_output, dec_state = self.rnn(emb, self.state["hidden"][0])
        else:
            rnn_output, dec_state = self.rnn(emb, self.state["hidden"])

        # Check
        tgt_len, tgt_batch, _ = tgt.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(tgt_len, output_len)
        aeq(tgt_batch, output_batch)

        # Calculate the attention.
        if not self.attentional:
            dec_outs = rnn_output
        else:
            dec_outs, p_attn = self.attn(
                rnn_output.transpose(0, 1).contiguous(),
                memory_bank.transpose(0, 1),
                memory_lengths=memory_lengths
            )
            attns["std"] = p_attn

        dec_outs = self.dropout(dec_outs)
        return dec_state, dec_outs, attns

    def _build_rnn(self, rnn_type, **kwargs):
        rnn, _ = rnn_factory(rnn_type, **kwargs)
        return rnn

    @property
    def _input_size(self):
        return self.emb_size

class InputFeedRNNDecoder(RNNDecoderBase):
    """Input feeding based decoder.
    """
    def _run_forward_pass(self, tgt, memory_bank, memory_lengths=None):
        # Additional args check.
        input_feed = self.state["input_feed"].squeeze(0)
        input_feed_batch, _ = input_feed.size()
        _, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        dec_outs = []
        attns = {}
        if self.attn is not None:
            attns["std"] = []

        emb = self._encode_positions(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        dec_state = self.state["hidden"]

        # Input feed concatenates hidden state with
        # input at every time step.
        for emb_t in emb.split(1):
            decoder_input = torch.cat([emb_t.squeeze(0), input_feed], 1)
            rnn_output, dec_state = self.rnn(decoder_input, dec_state)
            if self.attentional:
                decoder_output, p_attn, input_feed = self.attn(
                    rnn_output,
                    memory_bank.transpose(0, 1),
                    memory_lengths=memory_lengths)
                attns["std"].append(p_attn)
            else:
                decoder_output = rnn_output

            decoder_output = self.dropout(decoder_output)
            #input_feed = decoder_output

            dec_outs += [decoder_output]

        return dec_state, dec_outs, {"attns": attns}

    def _build_rnn(self, rnn_type, **kwargs):
        rnn, _ = rnn_factory(rnn_type, **kwargs)
        return rnn

    @property
    def _input_size(self):
        return self.emb_size + self.hidden_size

    def update_dropout(self, dropout):
        self.dropout.p = dropout
        self.rnn.dropout.p = dropout
