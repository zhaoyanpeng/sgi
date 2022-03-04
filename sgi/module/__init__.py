import math
from torch.nn.init import kaiming_uniform_, xavier_uniform_, xavier_normal_, normal_, constant_

#initializr = lambda x: xavier_normal_(x)
#initializr = lambda x: normal_(x, std=0.02)
#initializr = lambda x: kaiming_uniform_(x, a=math.sqrt(5))
#initializr = lambda x: xavier_uniform_(x)

from .sign import sign
from .sparsemax import sparsemax

from .global_attn import build_attention_head, ATTENTION_HEADS_REGISTRY, GlobalAttention
from .variational_attn import VariationalAttention

ATTENTION_HEADS_REGISTRY.register(VariationalAttention)


# embedders
from .embedder import *
from .mini_tf import *
from .rnn import *
# optimizer
from .lars import * 
# encoder heads
from .encoder import *
from .decoder import *
from .loss import *
