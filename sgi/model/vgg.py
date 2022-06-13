from omegaconf import OmegaConf
import math
import os, re
import torch
from torch import nn

import torch.distributed as dist
from torch.nn.parallel import data_parallel
import torchvision.models as models


class PretrainedVGG(nn.Module):
    def __init__(self, cfg, echo):
        super().__init__()
        self.cfg = cfg
        self.echo = echo

    def forward(self, objects, **kwargs):
        """ Batch-first images: (B, C, W, H).
        """
        features = self.encoder(objects)
        return None, (None, features) 

    def collect_state_dict(self):
        return (
            self.state_dict(), 
        )

    def stats(self): 
        return ""

    def reset(self):
        pass
    
    def reduce_grad(optim_rate, sync=False):
        raise NotImplementedError("Gradient Reduce")

    def report(self, gold_file=None):
        return ""

    def init_weights(self) -> None:
        pass

    def _build_encoder(self):
        vgg = getattr(models, self.cfg.model.name)(pretrained=True)
        vgg.classifier = vgg.classifier[:-1]
        model = vgg.train(False)
        self.echo(f"Loading pre-trained `{self.cfg.model.name}`")
        return model 
    
    def build(self, encoder_vocab, decoder_vocab):
        tunable_params = dict()

        self.decoder = None
        self.encoder = self._build_encoder()

        self.cuda(self.cfg.rank)
        return tunable_params
