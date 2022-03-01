import torch
import torch.nn as nn

from typing import Tuple
from torch import nn, Tensor
from collections import defaultdict

from sgi.util import Params, DistInfo
from .rnn_head import InputFeedRNNDecoder

class StdRNNDecoderHead(InputFeedRNNDecoder):
    def __init__(self, cfg, token_vocab):
        super(StdRNNDecoderHead, self).__init__(cfg, token_vocab)
        assert cfg.attention.name == "GlobalAttention"
        self.dist_type = "none"
        self.mode = "none"

class ViRNNDecoderHead(InputFeedRNNDecoder):
    def __init__(self, cfg, token_vocab):
        super(ViRNNDecoderHead, self).__init__(cfg, token_vocab)
        assert cfg.attention.name == "VariationalAttention"
        self.dist_type = self.attn.p_dist_type
        self.mode = self.attn.mode

    def _run_forward_pass(
        self, x, lengths, memory, memory_mask=None, enforce_sorted=False, q_scores=None
    ):
        dec_state = self.state["hidden"]
        dec_state = dec_state[0] if isinstance(self.encoder, nn.GRU) else dec_state
        input_feed = self.state["input_feed"]

        step_dim = 1 # as we have batch-first input
        attns = defaultdict(list)
        dist_infos = []
        baselines = []
        dec_outs = []

        for i, emb in enumerate(x.split(1, dim=step_dim)):
            emb = emb.squeeze(step_dim)
            decoder_input = torch.cat([emb, input_feed], 1).unsqueeze(step_dim)
            rnn_output, dec_state = self.encoder(decoder_input, dec_state)
            rnn_output = rnn_output.squeeze(step_dim)

            q_scores_i = None
            if q_scores is not None:
                q_scores_i = Params(
                    alpha=q_scores.alpha[i],
                    log_alpha=q_scores.log_alpha[i],
                    dist_type=q_scores.dist_type,
                )
                attns["q"].append(q_scores.alpha[i])

            decoder_output, p_attn, input_feed, baseline, dist_info = self.attn(
                rnn_output, memory, memory_mask=memory_mask, q_scores=q_scores_i
            )

            dec_outs.append(self.dropout(decoder_output))
            dist_infos.append(dist_info)
            attns["std"].append(p_attn)
            if baseline is not None: # from p(a | x, c)
                baselines.append(self.dropout(baseline))

        q_info = Params(
            alpha = q_scores.alpha,
            dist_type = q_scores.dist_type,
            samples = torch.stack([d.q.samples for d in dist_infos], dim=0)
                if dist_infos[0].q.samples is not None else None,
            log_alpha = q_scores.log_alpha,
            sample_log_probs = torch.stack([d.q.sample_log_probs for d in dist_infos], dim=0)
                if dist_infos[0].q.sample_log_probs is not None else None,
            sample_log_probs_q = torch.stack([d.q.sample_log_probs_q for d in dist_infos], dim=0)
                if dist_infos[0].q.sample_log_probs_q is not None else None,
            sample_log_probs_p = torch.stack([d.q.sample_log_probs_p for d in dist_infos], dim=0)
                if dist_infos[0].q.sample_log_probs_p is not None else None,
            sample_p_div_q_log = torch.stack([d.q.sample_p_div_q_log for d in dist_infos], dim=0)
                if dist_infos[0].q.sample_p_div_q_log is not None else None,
        ) if q_scores is not None else None
        p_info = Params(
            alpha = torch.stack([d.p.alpha for d in dist_infos], dim=0),
            dist_type = dist_infos[0].p.dist_type,
            log_alpha = torch.stack([d.p.log_alpha for d in dist_infos], dim=0)
                if dist_infos[0].p.log_alpha is not None else None,
            samples = torch.stack([d.p.samples for d in dist_infos], dim=0)
                if dist_infos[0].p.samples is not None else None,
        )
        dist_info = DistInfo(
            q=q_info,
            p=p_info,
        )
        return dec_state, dec_outs, attns, baselines, dist_info

    def _create_loss_state(self, output, target, dist_info=None, baselines=None):
        state = {"output": output, "target": target}
        if dist_info is not None:
            if dist_info.p is not None:
                state["p_samples"] = dist_info.p.samples
                if dist_info.p.dist_type == "categorical":
                    state["p_alpha"] = dist_info.p.alpha
                    state["p_log_alpha"] = dist_info.p.log_alpha
                else:
                    raise Exception("Unimplemented distribution")
            if dist_info.q is not None:
                state["q_samples"] = dist_info.q.samples
                if dist_info.q.dist_type == "categorical":
                    state["q_alpha"] = dist_info.q.alpha
                    state["q_log_alpha"] = dist_info.q.log_alpha
                    if self.mode != 'wsram':
                        state["q_sample_log_probs"] = dist_info.q.sample_log_probs
                    else:
                        state["sample_log_probs_q"] = dist_info.q.sample_log_probs_q
                        state["sample_log_probs_p"] = dist_info.q.sample_log_probs_p
                        state["sample_p_div_q_log"] = dist_info.q.sample_p_div_q_log
                else:
                    raise Exception("Unimplemented distribution")

        if baselines is not None:
            state["baselines"] = baselines
        return state

    def forward( 
        self, 
        x: Tensor,
        memory: Tensor=None,
        memo_attn_mask: Tensor=None,
        memo_key_padding_mask: Tensor=None,
        enforce_sorted: bool=False,
        q_scores: Tensor=None,
        **kwargs,
    ):
        assert x.dim() == 2, f"expect x of shape (B, L)"
        input_x = x[:, :-1] # causal LM sees the past
        x_embed = self._encode_positions(input_x)

        x_mask = x == self.token_vocab.PAD_IDX
        x_lengths = (~x_mask).sum(-1) - 1

        dec_state, dec_outs, attns, baselines, dist_info = self._run_forward_pass(
            x_embed, x_lengths, memory, q_scores=q_scores,
            memory_mask=memo_key_padding_mask, enforce_sorted=enforce_sorted,
        )

        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        self.state["hidden"] = dec_state

        # batch-second outputs
        if type(dec_outs) == list: # T x K x N x H
            dec_outs = torch.stack(dec_outs)
        for k in attns:
            if type(attns[k]) == list:
                attns[k] = torch.stack(attns[k])
        if len(baselines) > 0:
            baselines = torch.stack(baselines)
        else:
            baselines = None

        gold_x = x[:, 1:].transpose(0, 1).contiguous()
        #dec_outs = self.predictor(dec_outs)
        loss_state = self._create_loss_state(
            dec_outs, gold_x, baselines=baselines, dist_info=dist_info
        )
        return dec_outs, gold_x, {"loss_state": loss_state, "attns": attns}
