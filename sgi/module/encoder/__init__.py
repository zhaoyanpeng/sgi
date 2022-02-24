from .encoder_head import build_encoder_head, ENCODER_HEADS_REGISTRY

from .rnn_head import RNNEncoder
from .vi_head import BiEncInference

ENCODER_HEADS_REGISTRY.register(RNNEncoder)
ENCODER_HEADS_REGISTRY.register(BiEncInference)
