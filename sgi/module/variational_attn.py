import torch
import torch.nn as nn
import torch.nn.functional as F

from sgi.module import sparsemax
from sgi.util import Params, DistInfo

def sample_gumbel(x, K, eps=1e-20):
    N, T, S = x.shape[:3]
    noise = torch.rand((K, N, T, S)).to(x)
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    return noise

def gumbel_softmax_sample(log_probs, K, temperature):
    noise = sample_gumbel(log_probs, K) # K, N, T, S
    x = (log_probs.unsqueeze(0) + noise) / temperature
    x = F.softmax(x, dim=-1)
    return x.view_as(log_probs)

class VariationalAttention(nn.Module):
    def __init__(self, cfg, src_dim=None, tgt_dim=None, **kwargs):
        super(VariationalAttention, self).__init__()
        src_dim = src_dim or cfg.src_dim
        tgt_dim = tgt_dim or cfg.tgt_dim
        attn_type = cfg.attn_type
        attn_func = cfg.attn_func

        self.temperature = cfg.temperature
        self.p_dist_type = cfg.p_dist_type
        self.q_dist_tyqe = cfg.q_dist_type
        self.use_prior = cfg.use_prior
        self.nsample = cfg.nsample

        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.dim = cfg.attn_size

        self.mode = cfg.mode
        self.k = 0

        assert attn_type in ["dot", "general", "mlp"], (
            f"Please select a valid attention type (got {attn_type})."
        )
        self.attn_type = attn_type
        self.activation = nn.Tanh() if attn_type in ["general", "dot"] else nn.Identity()

        assert attn_func in ["softmax", "sparsemax"], (
            "Please select a valid attention function (got {attn_func})."
        )
        self.attn_func = F.softmax if attn_func == "softmax" else sparsemax

        if self.attn_type == "general":
            self.linear_in = nn.Linear(tgt_dim, src_dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_memory = nn.Linear(src_dim, self.dim, bias=False)
            self.linear_source = nn.Linear(tgt_dim, self.dim, bias=False)
            self.linear_in = nn.Linear(self.dim, 1, bias=False)

        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(src_dim + tgt_dim, tgt_dim, bias=out_bias)

    def score(self, h_t, h_s):
        """
        Args:
          h_t (FloatTensor): sequence of queries ``(batch, tgt_len, dim)``
          h_s (FloatTensor): sequence of sources ``(batch, src_len, dim``
        Returns:
          FloatTensor: raw attention scores (unnormalized) for each src index ``(batch, tgt_len, src_len)``
        """
        B, S, H_s = h_s.size()
        B, T, H_t = h_t.size()

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(B * T, H_t)
                h_t_ = self.linear_in(h_t_)
                h_t  = h_t_.view(B, T, H_s)
            h_s_ = h_s.transpose(1, 2)
            # (B, T, H) x (B, H, S) -> (B, T, S)
            return torch.bmm(h_t, h_s_)
        elif self.attn_type == "mlp":
            H = self.dim
            wq = self.linear_source(h_t.contiguous().view(-1, H_t))
            wq = wq.view(B, T, 1, H)
            wq = wq.expand(B, T, S, H)

            uh = self.linear_memory(h_s.contiguous().view(-1, H_s))
            uh = uh.view(B, 1, S, H)
            uh = uh.expand(B, T, S, H)

            # (B, T, S, H)
            wquh = torch.tanh(wq + uh)
            return self.linear_in(wquh.view(-1, H)).view(B, T, S)

    def sample_attn(self, params, nsample=1):
        dist_type = params.dist_type
        if dist_type == "categorical":
            alpha = params.alpha
            log_alpha = params.log_alpha
            K = nsample
            N, T, S = alpha.shape[:3]
            attns_id = torch.distributions.categorical.Categorical(
               alpha.view(N*T, S)
            ).sample(
                torch.Size([nsample])
            ).view(K, N, T, 1)
            attns = torch.Tensor(K, N, T, S).zero_().cuda()
            attns.scatter_(3, attns_id, 1)
            attns = attns.to(alpha)
            # log alpha: K, N, T, S
            log_alpha = log_alpha.unsqueeze(0).expand(K, N, T, S)
            sample_log_probs = log_alpha.gather(3, attns_id.to(log_alpha.device)).squeeze(3)
            return attns, sample_log_probs
        else:
            raise Exception("Unsupported dist")
        return attns, None

    def sample_attn_gumbel(self, params, temperature, nsample=1):
        dist_type = params.dist_type
        if dist_type == "categorical":
            alpha = params.alpha
            log_alpha = params.log_alpha
            K = nsample
            N, T, S = alpha.shape[:3]
            attns = gumbel_softmax_sample(log_alpha, K, temperature) # K, N, T, S
            # log alpha: K, N, T, S
            log_alpha = log_alpha.unsqueeze(0).expand(K, N, T, S)
            return attns, None 
        else:
            raise Exception("Unsupported dist")
        return attns, None

    def sample_attn_wsram(self, q_scores, p_scores, nsample=1):
        dist_type = q_scores.dist_type
        assert p_scores.dist_type == dist_type
        if dist_type == "categorical":
            alpha_q = q_scores.alpha
            log_alpha_q = q_scores.log_alpha
            K = nsample
            N, T, S = alpha.shape[:3]
            attns_id = torch.distributions.categorical.Categorical(
               alpha_q.view(N*T, S)
            ).sample(
                torch.Size([nsample])
            ).view(K, N, T, 1)
            attns = torch.Tensor(K, N, T, S).zero_().cuda()
            attns.scatter_(3, attns_id, 1)
            q_sample = attns.to(alpha_q)
            # log alpha: K, N, T, S
            log_alpha_q = log_alpha_q.unsqueeze(0).expand(K, N, T, S)
            sample_log_probs_q = log_alpha_q.gather(3, attns_id.to(log_alpha_q.device)).squeeze(3)
            log_alpha_p = p_scores.log_alpha
            log_alpha_p = log_alpha_p.unsqueeze(0).expand(K, N, T, S)
            sample_log_probs_p = log_alpha_p.gather(3, attns_id.to(log_alpha_p.device)).squeeze(3)
            sample_p_div_q_log = sample_log_probs_p - sample_log_probs_q
            return q_sample, sample_log_probs_q, sample_log_probs_p, sample_p_div_q_log
        else:
            raise Exception("Unsupported dist")

    def forward(self, source, memory, memory_mask=None, q_scores=None, **kwargs):
        one_step = False
        if source.dim() == 2:
            source = source.unsqueeze(1)
            one_step = True
            if q_scores is not None and q_scores.alpha is not None:
                q_scores = Params(
                    alpha=q_scores.alpha.unsqueeze(1),
                    log_alpha=q_scores.log_alpha.unsqueeze(1),
                    dist_type=q_scores.dist_type,
                )

        B, S, H_s = memory.size()
        B, T, H_t = source.size()

        # compute attention scores, as in Luong et al.
        # Params should be T x B x S
        if self.p_dist_type == "categorical":
            scores = self.score(source, memory)
            if memory_mask is not None:
                assert memory_mask.dim() == 2
                mask = memory_mask.unsqueeze(1)
                scores.data.masked_fill_(mask, float('-inf'))
            if self.k > 0 and self.k < scores.size(-1):
                topk, idx = scores.data.topk(self.k)
                new_attn_score = torch.zeros_like(scores.data).fill_(float("-inf"))
                new_attn_score = new_attn_score.scatter_(2, idx, topk)
                scores = new_attn_score
            log_scores = F.log_softmax(scores, dim=-1)
            scores = log_scores.exp()

            p_attn = scores

            p_scores = Params(
                alpha=scores,
                log_alpha=log_scores,
                dist_type=self.p_dist_type,
            )

        # soft attention under p, also the input feed
        p_ctx = torch.bmm(p_attn, memory) # (B, T, H_s)

        baseline = None # under p
        if self.mode != 'wsram': # baseline (B, T, H_t)
            source_c = torch.cat([source, p_ctx], -1)
            baseline = torch.tanh(self.linear_out(source_c))

        # q_attn: K x N x T x S
        q_sample, p_sample, sample_log_probs = None, None, None
        sample_log_probs_q, sample_log_probs_p, sample_p_div_q_log = None, None, None
        if self.mode == "sample":
            if q_scores is None or self.use_prior:
                p_sample, sample_log_probs = self.sample_attn(
                    p_scores, nsample=self.nsample,
                )
                q_attn = p_sample
            else:
                q_sample, sample_log_probs = self.sample_attn(
                    q_scores, nsample=self.nsample,
                )
                q_attn = q_sample
        elif self.mode == "gumbel":
            if q_scores is None or self.use_prior:
                p_sample, _ = self.sample_attn_gumbel(
                    p_scores, self.temperature, nsample=self.nsample,
                )
                q_attn = p_sample
            else:
                q_sample, _ = self.sample_attn_gumbel(
                    q_scores, self.temperature, nsample=self.nsample,
                )
                q_attn = q_sample
        elif self.mode == "wsram":
            assert q_scores is not None
            q_sample, sample_log_probs_q, sample_log_probs_p, sample_p_div_q_log = self.sample_attn_wsram(
                q_scores, p_scores, nsample=self.nsample,
            )
            q_attn = q_sample
        elif self.mode == "enum" or self.mode == "exact":
            q_attn = None


        if q_attn is not None:
            q_ctx = torch.bmm(
                q_attn.view(-1, T, S),
                memory.unsqueeze(0).repeat(self.nsample, 1, 1, 1).view(-1, S, H_s)
            ).view(self.nsample, B, T, H_s) # (K, B, T, H_s)
        else: # K == S for enumeration
            q_ctx = (
                memory.unsqueeze(0).repeat(T, 1, 1, 1).permute(2, 1, 0, 3)
            ) # (T, B, S, H_s) -> (S, B, T, H_s)
        source = source.unsqueeze(0).repeat(q_ctx.shape[0], 1, 1, 1)
        source_c = torch.cat([source, q_ctx], -1)
        attn_h = torch.tanh(self.linear_out(source_c))

        if one_step:
            if baseline is not None: # B x H
                baseline = baseline.squeeze(1)
            p_attn = p_attn.squeeze(1) # B x S
            p_ctx = p_ctx.squeeze(1)   # B x H_s
            attn_h = attn_h.squeeze(2) # K x N x H_t

            q_scores = Params(
                alpha = q_scores.alpha.squeeze(1) if q_scores.alpha is not None else None,
                dist_type = q_scores.dist_type,
                samples = q_sample.squeeze(2) if q_sample is not None else None,
                sample_log_probs = sample_log_probs.squeeze(2) if sample_log_probs is not None else None,
                sample_log_probs_q = sample_log_probs_q.squeeze(2) if sample_log_probs_q is not None else None,
                sample_log_probs_p = sample_log_probs_p.squeeze(2) if sample_log_probs_p is not None else None,
                sample_p_div_q_log = sample_p_div_q_log.squeeze(2) if sample_p_div_q_log is not None else None,
            ) if q_scores is not None else None
            p_scores = Params(
                alpha = p_scores.alpha.squeeze(1),
                log_alpha = log_scores.squeeze(1),
                dist_type = p_scores.dist_type,
                samples = p_sample.squeeze(2) if p_sample is not None else None,
            )
        else:
            raise Exception("`multi-step' not supported yet")

        # For now, don't include samples.
        dist_info = DistInfo(
            q = q_scores,
            p = p_scores,
        )

        # attn_h: output features for prediction
        #   either K x N x H, or T x K x N x H
        return attn_h, p_attn, p_ctx, baseline, dist_info
