import numpy as np
import os, sys, time
import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from fvcore.common.registry import Registry

LOSS_HEADS_REGISTRY = Registry("LOSS_HEADS")
LOSS_HEADS_REGISTRY.__doc__ = """
Registry for encoder heads.
"""

initializr = lambda x: None 

def build_loss_head(cfg, vocab):
    return LOSS_HEADS_REGISTRY.get(cfg.name)(cfg, vocab)

@LOSS_HEADS_REGISTRY.register()
class DummyLossHead(nn.Module):
    def __init__(self, cfg, token_vocab):
        super().__init__()
        pass
    def _reset_parameters(self):
        pass
    def output_shape(self):
        return 0 
    def forward(self, *args, **kwargs):
        return None, None 

class LossHead(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def infer(self):
        pass
    def report(self):
        return ""

@LOSS_HEADS_REGISTRY.register()
class LMLossHead(LossHead):
    def __init__(self, cfg, token_vocab, **kwargs):
        super().__init__()
        self.token_vocab = token_vocab
        self.logit_scale = (
            nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) if cfg.scaling else
            torch.ones([], requires_grad=False) * np.log(1 / 1)
        )
        self.loss_fn = nn.CrossEntropyLoss(
            reduction="none", ignore_index=self.token_vocab.PAD_IDX
        )
        self.add_dummy = cfg.add_dummy
        self.cate_type = cfg.cate_type
        self.optim_only_relation = cfg.optim_only_relation 
        relation_words = ["left", "right", "front", "behind"]
        self.relation_words = {
            word: self.token_vocab(word) for word in relation_words
        }
        self.accuracies = {word: [0] * 2 for word in relation_words + ["overall"]}
        self.reduce = False 

    def select(self, logit, target):
        if self.cate_type == "atomic_object":
            sel_logit = logit[:, 2::3]
            sel_target = target[:, 2::3]
        elif "_oor" in self.cate_type:
            sel_logit = logit[:, 8::9]
            sel_target = target[:, 8::9]
        elif "_oro" in self.cate_type:
            sel_logit = logit[:, 4::9]
            sel_target = target[:, 4::9]
        else:
            sel_logit = logit
            sel_target = target
        
        def print_seq(x):
            xx = x.cpu().tolist()
            for x in xx:
                print(self.token_vocab(x), len(x))
        #print_seq(target)
        #print_seq(sel_target)
        #import sys; sys.exit(0)

        return sel_logit, sel_target

    def report(self, gold_file=None):
        # compute accuracies, called every epoch
        result = " ".join(
            [f"{k}: {(v[0] / v[1]) * 100:.3f} ({v[1]})" for k, v in self.accuracies.items()]
        )
        self.accuracies = {k: [0] * 2 for k, _ in self.accuracies.items()} # reset
        return result 

    def _estimate_loss(self, logits, x2):
        losses = self.loss_fn(
            logits.reshape(-1, logits.shape[-1]), x2.reshape(-1)
        ) 
        losses = losses.view(x2.size())

        loss_sum = losses.sum() 
        ntoken = (x2 != self.token_vocab.PAD_IDX).sum()
        loss = loss_sum / ntoken
        return loss, (ntoken, losses)

    def infer(self, x1, x2, *args, **kwargs): 
        x1_new, x2_new = self.select(x1, x2)
        results = self._estimate_loss(x1_new, x2_new)
        # overall and relation-specific loss
        x1 = x1.argmax(dim=-1).reshape(-1)
        x2 = x2.reshape(-1)
        mask = x2 != self.token_vocab.PAD_IDX
        for word, metric in self.accuracies.items():
            if word in self.relation_words: # overall loss
                wid = self.relation_words[word]
                gold = (x2 == wid) * mask
                pred = (x1 == wid) * gold
                metric[0] += pred.sum()
                metric[1] += gold.sum()
            else:
                metric[0] += ((x1 == x2) * mask).sum() 
                metric[1] += mask.sum()
        return results 

    def forward(self, x1, x2, *args, **kwargs):
        if not self.training:
            return self.infer(x1, x2, *args, **kwargs)
        logits = self.logit_scale.exp() * x1
        if self.optim_only_relation and self.training:
            logits, x2 = self.select(logits, x2)
        return self._estimate_loss(logits, x2)
