import math
from torch.nn.init import kaiming_uniform_, xavier_uniform_, xavier_normal_, normal_, constant_

#initializr = lambda x: xavier_normal_(x)
#initializr = lambda x: normal_(x, std=0.02)
#initializr = lambda x: kaiming_uniform_(x, a=math.sqrt(5))
#initializr = lambda x: xavier_uniform_(x)

# embedders
from .embedder import *
from .mini_tf import *
# optimizer
from .lars import * 
# encoder heads
from .encoder import *
from .decoder import *
from .loss import *
