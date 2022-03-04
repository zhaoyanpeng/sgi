import numpy as np
import os, sys, time, math
import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

from sgi.util import Statistics

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
        self.ignore_index = self.token_vocab.PAD_IDX
        self.loss_fn = nn.CrossEntropyLoss(
            reduction="none", ignore_index=self.ignore_index
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
        if self.cate_type == "1_atomic_object":
            sel_logit = logit[:, 2::3]
            sel_target = target[:, 2::3]
        elif "1_oor" in self.cate_type:
            sel_logit = logit[:, 8::9]
            sel_target = target[:, 8::9]
        elif "1_oro" in self.cate_type:
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
            ["REL:"] + [f"{k}: {(v[0] / v[1]) * 100:.3f} ({v[1]})" for k, v in self.accuracies.items()]
        )
        self.accuracies = {k: [0] * 2 for k, _ in self.accuracies.items()} # reset
        return result 

    def _estimate_loss(self, logits, x2):
        losses = self.loss_fn(
            logits.reshape(-1, logits.shape[-1]), x2.reshape(-1)
        ) 
        losses = losses.view(x2.size())

        loss_sum = losses.sum() 
        ntoken = (x2 != self.ignore_index).sum()
        loss = (loss_sum / ntoken) if ntoken > 0 else loss_sum
        return loss, (ntoken, losses)

    def infer(self, x1, x2, *args, **kwargs): 
        x1_new, x2_new = self.select(x1, x2)
        results = self._estimate_loss(x1_new, x2_new)
        # overall and relation-specific loss
        x1 = x1.argmax(dim=-1).reshape(-1)
        x2 = x2.reshape(-1)
        mask = x2 != self.ignore_index
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

@LOSS_HEADS_REGISTRY.register()
class MLMLossHead(LMLossHead):
    def __init__(self, cfg, token_vocab, **kwargs):
        super().__init__(cfg, token_vocab, **kwargs)
        self.ignore_index = -100
        self.loss_fn = nn.CrossEntropyLoss(
            reduction="none", ignore_index=self.ignore_index
        )

@LOSS_HEADS_REGISTRY.register()
class ViLMLossHead(LMLossHead):
    def __init__(self, cfg, token_vocab, **kwargs):
        super().__init__(cfg, token_vocab, **kwargs)
        self.loss_fn = nn.NLLLoss(
            reduction="sum", ignore_index=self.token_vocab.PAD_IDX
        )
        self.train_baseline = False
        self.confidence = 1
        self.cate_type = ""
        self.one_hot = None
        self._alpha = 1

    def infer(self, x1, x2, *args, **kwargs):
        x1_new, x2_new = self.select(x1, x2)
        #results = self._estimate_loss(x1_new, x2_new)
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
        return None

    def forward(
        self, x1, x2, *args, loss_state=None, decoder=None, **kwargs
    ):
        assert decoder is not None, f"`decoder' should not be None."
        #if not self.training:
        #    return self.infer(decoder=decoder, **loss_state)
        #return self._estimate_loss(decoder=decoder, **loss_state)
        return self._compute_loss(decoder=decoder, **loss_state)

    def _decode(self, x, decoder, log_pa=None, pa=None):
        # log_pa: T x K x B x S # (T, K, B, V)
        scores = decoder.predictor(x).log_softmax(dim=-1)
        if x.dim() == 3: # short-circuit
            return scores
        if scores.size(1) == 1: # single sample
            scores = scores.squeeze(1)
        else:
            if decoder.mode == "exact" and log_pa is not None: # for exact marginal over p
                scores = scores + log_pa.transpose(1,2).unsqueeze(-1)
                scores = scores.logsumexp(dim=1, keepdim=False)
            elif decoder.mode == "enum" and pa is not None: # for exact elbo over q
                scores = scores * pa.transpose(1,2).unsqueeze(-1)
                scores = scores.sum(dim=1, keepdim=False)
            elif decoder.mode == "wsram":
                return scores
            else: # multiple samples w/ empirical mean
                scores = scores.logsumexp(dim=1, keepdim=False)
                scores = scores - math.log(x.size(1))
        return scores

    def _compute_loss(
        self, decoder=None,
        output=None, target=None,
        p_samples=None, q_samples=None,
        p_alpha=None, q_alpha=None,
        q_log_alpha=None,
        p_log_alpha=None,
        q_sample_log_probs=None,
        baselines=None,
        sample_log_probs_q=None,
        sample_log_probs_p=None,
        sample_p_div_q_log=None,
        dist_type=None,
        **kwargs,
    ):
        if decoder.mode in ["enum", "exact", "wsram", "gumbel"]:
            baselines = None

        # Reconstruction
        # TODO(jchiu): hacky, want to set use_prior.
        scores = self._decode(
            output, decoder,
            log_pa = q_log_alpha if q_log_alpha is not None else p_log_alpha,
            pa = q_alpha if q_alpha is not None else p_alpha,
        )
        if decoder.mode == 'wsram':
            log_p_y = scores # T, K, batch, S
            T, K, B, _ = log_p_y.size()
            #p_y = log_p_y.exp()
            log_p_y_sample = log_p_y.gather(3, target.unsqueeze(1).unsqueeze(-1)
                                            .expand(T, K, B, 1)).squeeze(3)
            w_unnormalized = (sample_p_div_q_log + log_p_y_sample).exp() #T, K, B
            w_normalized = w_unnormalized / w_unnormalized.sum(dim=1, keepdim=True)
            #bp = sample_p_div_q_log.exp()
            #bp = bp / bp.sum(dim=1, keepdim=True)
            #bq = 1. / K
            bp = 0
            bq = 0
            target_expand = target.unsqueeze(1).expand(T, K, B).contiguous().view(-1)
            # loss 1: w * log p (y)
            loss1 = - w_normalized.detach() * log_p_y_sample
            loss1 = loss1.view(-1)[target_expand.ne(self.token_vocab.PAD_IDX)].sum()
            # loss 2: (w - bp) * log p(a)
            loss2 = - (w_normalized - bp).detach() * sample_log_probs_p
            loss2 = loss2.view(-1)[target_expand.ne(self.token_vocab.PAD_IDX)].sum()
            # loss 3: (w - bq) log q a
            loss3 = - (w_normalized - bq).detach() * sample_log_probs_q
            loss3 = loss3.view(-1)[target_expand.ne(self.token_vocab.PAD_IDX)].sum()
            loss = loss1+loss2+loss3

            gtruth = target.view(-1)
            q_alpha = q_alpha.contiguous().view(-1, q_alpha.size(2))
            q_alpha = q_alpha[gtruth.ne(self.token_vocab.PAD_IDX)]
            p_alpha = p_alpha.contiguous().view(-1, p_alpha.size(2))
            p_alpha = p_alpha[gtruth.ne(self.token_vocab.PAD_IDX)]
            if decoder.dist_type == 'categorical':
                q = Categorical(q_alpha)
                p = Categorical(p_alpha)
            else:
                assert (False)
            kl = kl_divergence(q, p).sum()
            kl_data = kl.data

            scores_first = log_p_y[:,0,:,:]
            scores_first = scores_first.contiguous().view(-1, scores_first.size(-1))
            xent = self.loss_fn(scores_first, gtruth)
            xent_data = xent.data


            # loss per token
            ntoken = (gtruth != self.token_vocab.PAD_IDX).sum()
            loss_mean = loss / ntoken # average over tokens

            # inference
            prediction = scores_first.max(1)[1]
            non_padding = gtruth.ne(self.token_vocab.PAD_IDX)
            num_correct = prediction.eq(gtruth).masked_select(non_padding).sum()

            stats = Statistics(
                xent.item(), kl.item(), non_padding.sum().item(), num_correct.item()
            )

            if not self.training: # TODO hacky relation words
                self.infer(scores_first, gtruth)

            return loss, (ntoken, stats)

        scores = scores.view(-1, scores.size(-1))
        if baselines is not None:
            baselines = baselines.unsqueeze(1)
            scores_baseline = self._decode(baselines, decoder)
            scores_baseline = scores_baseline.view(-1, scores.size(-1))

        gtruth = target.view(-1)
        if self.confidence < 1:
            tdata = gtruth.data
            mask = torch.nonzero(tdata.eq(self.token_vocab.PAD_IDX)).squeeze()
            log_likelihood = torch.gather(scores.data, 1, tdata.unsqueeze(1))
            tmp_ = self.one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            if mask.dim() > 0:
                log_likelihood.index_fill_(0, mask, 0)
                tmp_.index_fill_(0, mask, 0)
            gtruth = Variable(tmp_, requires_grad=False)

        xent = self.loss_fn(scores, gtruth)
        if baselines is not None:
            xent_baseline = self.loss_fn(scores_baseline, gtruth)

        if q_sample_log_probs is not None and baselines is not None:
            # This code doesn't handle multiple samples
            scores_nopad = scores[gtruth.ne(self.token_vocab.PAD_IDX)]
            scores_baseline_nopad = scores_baseline[gtruth.ne(self.token_vocab.PAD_IDX)]
            gtruth_nopad = gtruth[gtruth.ne(self.token_vocab.PAD_IDX)]
            llh_ind = scores_nopad.gather(1, gtruth_nopad.unsqueeze(1))
            llh_baseline_ind = scores_baseline_nopad.gather(1, gtruth_nopad.unsqueeze(1))
            reward = (llh_ind.detach() - llh_baseline_ind.detach()).view(-1) # T*N
            q_sample_log_probs = q_sample_log_probs.view(-1) # T, N
            q_sample_log_probs = q_sample_log_probs[gtruth.ne(self.token_vocab.PAD_IDX)]
        if self.confidence < 1:
            # Default: report smoothed ppl.
            # loss_data = -log_likelihood.sum(0)
            xent_data = xent.data.clone()
        else:
            xent_data = xent.data.clone()

        # KL
        if q_alpha is not None:
            q_alpha = q_alpha.contiguous().view(-1, q_alpha.size(2))
            q_alpha = q_alpha[gtruth.ne(self.token_vocab.PAD_IDX)]
            p_alpha = p_alpha.contiguous().view(-1, p_alpha.size(2))
            p_alpha = p_alpha[gtruth.ne(self.token_vocab.PAD_IDX)]
            if decoder.dist_type == 'categorical':
                q = Categorical(q_alpha)
                p = Categorical(p_alpha)
            else:
                assert False, f"dist_type: {decoder.dist_type}"
            kl = kl_divergence(q, p).sum()
            loss = xent + self._alpha * kl
        else:
            kl = torch.zeros(1).to(xent)
            loss = xent

        # subtract reward
        if decoder.mode == 'gumbel':
            assert q_sample_log_probs is None
        if q_sample_log_probs is not None:
            loss = loss - (reward * q_sample_log_probs).sum()
            if self.train_baseline:
                loss = loss + xent_baseline

        # loss per token
        ntoken = (gtruth != self.token_vocab.PAD_IDX).sum()
        loss_mean = loss / ntoken # average over tokens

        # inference
        prediction = scores.max(1)[1]
        non_padding = gtruth.ne(self.token_vocab.PAD_IDX)
        num_correct = prediction.eq(gtruth).masked_select(non_padding).sum()

        stats = Statistics(
            xent.item(), kl.item(), non_padding.sum().item(), num_correct.item()
        )

        if not self.training: # TODO hacky relation words
            self.infer(scores, gtruth)

        return loss_mean, (ntoken, stats)
