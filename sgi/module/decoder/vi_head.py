import torch
import torch.nn as nn

from typing import Tuple
from torch import nn, Tensor
from collections import defaultdict

from sgi.util import aeq, Params, DistInfo
from .. import GlobalAttention, VariationalAttention
from .rnn_head import InputFeedRNNDecoder

class ViRNNDecoderHead(InputFeedRNNDecoder):
    def __init__(self, cfg, token_vocab):
        super(ViRNNDecoderHead, self).__init__(cfg, token_vocab, skip_attn=True)
        cfg = cfg.attention
        assert cfg is not None, f"the module has to be attentional but attention configs are missing"
        self.mode = cfg.mode
        self.dist_type = cfg.p_dist_type
        self.attn = VariationalAttention(
            src_dim         = cfg.src_dim,
            tgt_dim         = cfg.tgt_dim,
            attn_size       = cfg.attn_size,
            temperature     = cfg.temperature,
            p_dist_type     = cfg.p_dist_type,
            q_dist_type     = cfg.q_dist_type,
            use_prior       = cfg.use_prior,
            attn_type       = cfg.attn_type,
            attn_func       = cfg.attn_func,
            nsample         = cfg.nsample,
            mode            = cfg.mode,
        )

    def _run_forward_pass(self, tgt, memory_bank, memory_lengths=None, q_scores=None):
        input_feed = self.state["input_feed"].squeeze(0)
        dec_state = self.state["hidden"]

        attns = defaultdict(list)
        dist_infos = []
        baselines = []
        dec_outs = []

        step_dim = 1 # as we have batch-first input

        # input feed concatenates hidden state with input at every time step.
        for i, emb_t in enumerate(tgt.split(1, dim=step_dim)):
            decoder_input = torch.cat(
                [emb_t.squeeze(step_dim), input_feed], 1
            ).unsqueeze(0)
            rnn_output, dec_state = self.rnn(decoder_input, dec_state)
            rnn_output = rnn_output.squeeze(0) # remove length dim
            
            q_scores_i = None
            if q_scores is not None:
                q_scores_i = Params(
                    alpha=q_scores.alpha[i],
                    log_alpha=q_scores.log_alpha[i],
                    dist_type=q_scores.dist_type,
                )
                attns["q"].append(q_scores.alpha[i])

            # cross-attn
            if self.attentional:
                decoder_output, p_attn, input_feed, baseline, dist_info = self.attn(
                    rnn_output, memory_bank, memory_lengths=memory_lengths, q_scores=q_scores_i
                )
                attns["std"].append(p_attn)
                dist_infos.append(dist_info)
            else:
                decoder_output = rnn_output

            decoder_output = self.dropout(decoder_output)
            dec_outs.append(decoder_output)

            if baseline is not None: # from p(a | x, c)
                baseline = self.dropout(baseline)
                baselines.append(baseline)

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
        return dec_state, input_feed, dec_outs, attns, baselines, dist_info

    def create_loss_state(self, output, target, dist_info=None, baselines=None):
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
        tgt: Tensor,
        memory: Tensor=None,
        memo_attn_mask: Tensor=None,
        memo_key_padding_mask: Tensor=None,
        q_scores=None,
        **kwargs,
    ):
        assert tgt.dim() == 2, f"expect tgt of shape (B, L)"
        gold_tgt = tgt[:, 1:].transpose(0, 1).contiguous()
        tgt = tgt[:, :-1] # see the past
        tgt = self._encode_positions(tgt)

        memory_lengths = (
            None if memo_key_padding_mask is None 
            else (~memo_key_padding_mask).sum(-1) 
        )

        # run the forward pass of the RNN
        dec_state, input_feed, dec_outs, attns, baselines, dist_info = \
            self._run_forward_pass(
                tgt, memory, memory_lengths=memory_lengths, q_scores=q_scores
            )

        # concatenates sequence of tensors along a new dimension
        dec_outs = torch.stack(dec_outs, dim=0) # T x K x N x H
        if len(baselines) > 0:
            baselines = torch.stack(baselines, dim=0)
        else:
            baselines = None
        for k in attns:
            attns[k] = torch.stack(attns[k])

        # update the state with the result
        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)

        loss_state = self.create_loss_state(
            dec_outs, gold_tgt, dist_info=dist_info, baselines=baselines
        )

        return dec_outs, gold_tgt, {"loss_state": loss_state, "attns": attns}


class StdRNNDecoderHead(InputFeedRNNDecoder):
    def __init__(self, cfg, token_vocab):
        super(StdRNNDecoderHead, self).__init__(cfg, token_vocab, skip_attn=True)
        cfg = cfg.attention
        assert cfg is not None, f"the module has to be attentional but attention configs are missing"
        self.mode = "none"
        self.attn = GlobalAttention(
            cfg.src_dim, cfg.tgt_dim, cfg.attn_size,
            self.hidden_size, attn_type=cfg.attn_type, attn_func=cfg.attn_func
        )

    def _run_forward_pass(self, tgt, memory_bank, memory_lengths=None):
        input_feed = self.state["input_feed"].squeeze(0)
        dec_state = self.state["hidden"]

        attns = defaultdict(list)
        dec_outs = []

        step_dim = 1 # as we have batch-first input

        # input feed concatenates hidden state with input at every time step.
        for i, emb_t in enumerate(tgt.split(1, dim=step_dim)):
            decoder_input = torch.cat(
                [emb_t.squeeze(step_dim), input_feed], 1
            ).unsqueeze(0)
            rnn_output, dec_state = self.rnn(decoder_input, dec_state)
            rnn_output = rnn_output.squeeze(0) # remove length dim

            if self.attentional:
                decoder_output, p_attn, input_feed = self.attn(
                    rnn_output, memory_bank, memory_lengths=memory_lengths
                )
                attns["std"].append(p_attn)
            else:
                decoder_output = rnn_output

            decoder_output = self.dropout(decoder_output)
            #input_feed = decoder_output

            dec_outs += [decoder_output]

        return dec_state, input_feed, dec_outs, attns

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor=None,
        memo_attn_mask: Tensor=None,
        memo_key_padding_mask: Tensor=None,
        **kwargs,
    ):
        assert tgt.dim() == 2, f"expect tgt of shape (B, L)"
        gold_tgt = tgt[:, 1:].transpose(0, 1).contiguous()
        tgt = tgt[:, :-1] # see the past
        tgt = self._encode_positions(tgt)

        memory_lengths = (
            None if memo_key_padding_mask is None
            else (~memo_key_padding_mask).sum(-1)
        )

        # run the forward pass of the RNN
        dec_state, input_feed, dec_outs, attns = \
            self._run_forward_pass(
                tgt, memory, memory_lengths=memory_lengths
            )

        # concatenates sequence of tensors along a new dimension
        dec_outs = torch.stack(dec_outs, dim=0) # T x K x N x H
        for k in attns:
            attns[k] = torch.stack(attns[k])

        # update the state with the result
        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)

        loss_state = {"output": dec_outs, "target": gold_tgt}

        return dec_outs, gold_tgt, {"loss_state": loss_state, "attns": attns}
