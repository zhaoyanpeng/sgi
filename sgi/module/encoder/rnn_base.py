import torch.nn as nn

from .. import rnn_factory
from .encoder_head import MetaEncHead


class RNNEncoderBase(MetaEncHead):
    def __init__(self, cfg, token_vocab, bidirectional=False):
        super(RNNEncoderBase, self).__init__(cfg, token_vocab)
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

    @property
    def _input_size(self):
        return self.emb_size

    @property
    def output_size(self):
        k = 2 if self.bidirectional else 1
        return self.hidden_size * k

    def _build_rnn(self, rnn_type, **kwargs):
        rnn, _ = rnn_factory(rnn_type, **kwargs)
        return rnn

