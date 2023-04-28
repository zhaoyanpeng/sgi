from .decoder_head import build_decoder_head, DECODER_HEADS_REGISTRY

from .rnn_head import RNNDecoderBase, StdRNNDecoder, InputFeedRNNDecoder
from .sgi_head import RouteMiniTFDecHead, SGIMiniTFMLMDecHead, SGIMiniTFMLMLMDecHead
from .vi_head import StdRNNDecoderHead, ViRNNDecoderHead
from .birnn_head import (
    CatThenAttendRNNDecoderHead, AttendThenCatRNNDecoderHead, BiInputFeedRNNDecoderHead
)

DECODER_HEADS_REGISTRY.register(RouteMiniTFDecHead)
DECODER_HEADS_REGISTRY.register(SGIMiniTFMLMDecHead)
DECODER_HEADS_REGISTRY.register(SGIMiniTFMLMLMDecHead)

DECODER_HEADS_REGISTRY.register(StdRNNDecoder)
DECODER_HEADS_REGISTRY.register(InputFeedRNNDecoder)
DECODER_HEADS_REGISTRY.register(ViRNNDecoderHead)
DECODER_HEADS_REGISTRY.register(StdRNNDecoderHead)
DECODER_HEADS_REGISTRY.register(CatThenAttendRNNDecoderHead)
DECODER_HEADS_REGISTRY.register(AttendThenCatRNNDecoderHead)
DECODER_HEADS_REGISTRY.register(BiInputFeedRNNDecoderHead)
