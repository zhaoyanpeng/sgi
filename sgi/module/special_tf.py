from typing import Callable, Dict, List, Optional, Tuple, Union
import torch.nn.functional as F
from torch import nn, Tensor
import torch
import random

from . import MetaModule, MiniTFBlock, MiniTFAttention

__all__ = [
    "RouteTFBlock",
    "SpecialTFBlock",
    "SpecialTFAttention",
]


class RouteTFBlock(MiniTFBlock):
    def __init__(
        self, *args,
        self_ctx_dropout: float = .0,
        memo_ctx_dropout: float = .0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.self_ctx_dp = nn.Dropout(self_ctx_dropout)
        self.memo_ctx_dp = nn.Dropout(memo_ctx_dropout)

    def forward(
        self, q: Tensor,
        kv: Tensor = None,
        self_attn_mask: Tensor = None,
        self_key_padding_mask: Tensor = None,
        memory: Tensor = None,
        memo_attn_mask: Tensor = None,
        memo_key_padding_mask: Tensor = None,
        **kwargs
    ):
        if kv is None:
            k = v = q
        else:
            k = v = kv
        residual = q
        x, intra_attn_weight = self.intra_attn(
            q, k, v,
            attn_mask = self_attn_mask,
            key_padding_mask = self_key_padding_mask,
            **kwargs
        )
        x = self.intra_attn_ln(residual + self.intra_attn_dp(x))

        inter_attn_weight = None
        if self.inter_attn is not None:
            k = v = memory
            residual = q = x
            x, inter_attn_weight = self.inter_attn(
                q, k, v,
                attn_mask = memo_attn_mask,
                key_padding_mask = memo_key_padding_mask,
                **kwargs
            )
            alpha = 1.
            ## the standard
            #x = self.inter_attn_ln(residual + self.inter_attn_dp(x))
            ## dropout -> ab
            alpha = .0 if not self.training else 1.

            x = x * alpha # TODO simply skip inter attn at test time
            x = self.inter_attn_ln(self.self_ctx_dp(residual) + self.memo_ctx_dp(x))

        x = self.ff_ln(x + self.ff_dp(self.ff(x)))
        return x, (intra_attn_weight, inter_attn_weight)


class SpecialTFBlock(MiniTFBlock):
    def __init__(
        self, *args,
        self_ctx_dropout: float = .0,
        memo_ctx_dropout: float = .0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.self_ctx_dp = nn.Dropout(self_ctx_dropout)
        self.memo_ctx_dp = nn.Dropout(memo_ctx_dropout)

        D = self.intra_attn.D

        self.self_ctx_ln = nn.LayerNorm(D)
        self.memo_ctx_ln = nn.LayerNorm(D)

        self.beta_slice = 8
        self.beta_alpha = torch.tensor([2.])
        self.beta_sampler = torch.distributions.beta.Beta(self.beta_alpha, self.beta_alpha)

    def forward(
        self, q: Tensor,
        kv: Tensor = None,
        self_attn_mask: Tensor = None,
        self_key_padding_mask: Tensor = None,
        memory: Tensor = None,
        memo_attn_mask: Tensor = None,
        memo_key_padding_mask: Tensor = None,
        epoch_beta: float = 0.,
        split_vl: bool = False, # split vision and language
        **kwargs
    ):
        if kv is None:
            k = v = q
        else:
            k = v = kv
        residual = q 
        x, intra_attn_weight = self.intra_attn(
            q, k, v, 
            attn_mask = self_attn_mask, 
            key_padding_mask = self_key_padding_mask, 
            **kwargs
        ) 
        x = self.intra_attn_ln(residual + self.intra_attn_dp(x))
        
        inter_attn_weight = pair_logit = None
        if self.inter_attn is not None:
            k = v = memory
            residual = q = x
            x, inter_attn_weight = self.inter_attn(
                q, k, v, 
                attn_mask = memo_attn_mask,
                key_padding_mask = memo_key_padding_mask, 
                **kwargs
            ) 
            dim = 1 if self.inter_attn.mode != 2 else 2
            #dim = 1 if x.shape[1] != residual.shape[1] else 2 # FIXME sentence length and pair number may be equal
            x = x.unsqueeze(dim)

            alpha = .0 if not split_vl and not self.training else 1.

            #alpha, x = 1., x * .0

            stack_first = True
            if split_vl and stack_first:
                alpha = .0
                v, l = x, residual.unsqueeze(2)
                v = v.expand(-1, l.shape[1], -1, -1)
                x = torch.cat([l, v], dim=2)

            ## the standard
            residual = residual.unsqueeze(2) * alpha
            ## annealing
            #residual = residual.unsqueeze(2) * epoch_beta
            ## zero out
            #residual = torch.zeros_like(residual) # reset to zero

            zero, p_zero = False, .15 # zero out w/ a probability
            if zero and random.random() < p_zero and self.training:
                residual = residual * 0.

            if not split_vl or stack_first:
                ## the standard
                #x = self.inter_attn_ln(residual + self.inter_attn_dp(x))
                ## dropout -> b
                #x = self.inter_attn_ln(residual + F.dropout(x, p=0.15, training=self.training))
                ## dropout -> ab
                #x = self.inter_attn_ln(F.dropout(residual, p=0.15, training=self.training) + F.dropout(x, p=0.15, training=self.training))
                ## dropout -> a
                #x = self.inter_attn_ln(F.dropout(residual, p=0.15, training=self.training) + x)
                ## dropout -> ab
                x = self.inter_attn_ln(self.self_ctx_dp(residual) + self.memo_ctx_dp(x))

                ## lndp
                #x = (
                #    self.self_ctx_dp(self.self_ctx_ln(residual)) +
                #    self.memo_ctx_dp(self.memo_ctx_ln(x))
                #)

                ## lndpln
                #x = self.inter_attn_ln(
                #    self.self_ctx_dp(self.self_ctx_ln(residual)) +
                #    self.memo_ctx_dp(self.memo_ctx_ln(x))
                #)

                #shape = list(residual.shape)
                #shape[-1] = shape[-1] // self.beta_slice
                #beta = self.beta_sampler.sample(shape).squeeze(-1)
                #beta = beta.to(residual).repeat_interleave(self.beta_slice, -1)

                ## lndp - beta
                #x = (
                #    (1 - beta) * self.memo_ctx_ln(x) +
                #    beta * self.self_ctx_ln(residual)
                #)

                ## lndpln - beta
                #x = self.inter_attn_ln(
                #    (1 - beta) * self.memo_ctx_ln(x) +
                #    beta * self.self_ctx_ln(residual)
                #)
            else:
                v = self.inter_attn_ln(self.memo_ctx_dp(x))
                l = self.inter_attn_ln(self.self_ctx_dp(residual))

        if not split_vl or stack_first:
            ## the standard
            x = self.ff_ln(x + self.ff_dp(self.ff(x)))
            ## drop it out
            #x = self.ff_ln(x + F.dropout(self.ff(x), p=0.15, training=self.training))
        else:
            v = self.ff_ln(v + self.ff_dp(self.ff(v)))
            l = self.ff_ln(l + self.ff_dp(self.ff(l)))

            v = v.expand(-1, l.shape[1], -1, -1)
            x = torch.cat([l, v], dim=2)

        return x, (intra_attn_weight, inter_attn_weight)


class SpecialTFAttention(MiniTFAttention):
    def __init__(
        self,
        D: int,
        N: int,
        kdim: int = None,
        vdim: int = None,
        bias: bool = True,
        qk_scale: float = None,
        attn_dropout: float = .0,
        proj_dropout: float = .0,
        routing: bool = True, # dynamic routing, do not use in the last layer
        **kwargs,
    ):
        super().__init__(
            D, N, attn_dropout=attn_dropout, proj_dropout=proj_dropout, **kwargs,
        )

        self.shared_k = True
        self.routing = routing

        self.mode = 1
        if self.mode in {1, 2}:
            self.proj = nn.Linear(D * 2, D)

    def forward(
        self, 
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor = None,
        key_padding_mask: Tensor = None,
        **kwargs
    ):
        if q.data_ptr() == k.data_ptr() == v.data_ptr():
            k, v, q = self._proj_qkv(q)
        elif k.data_ptr() == v.data_ptr():
            k, v = self._proj_kv(k)
            q = self._proj_q(q) 
        else:
            q = self._proj_q(q)
            k = self._proj_k(k)
            v = self._proj_v(v)

        B, T, S = q.shape[0], q.shape[1], k.shape[1]
        
        # (B, L, D) -> (B, L, N, H) -> (B, N, L, H)
        q = q.contiguous().reshape(B, T, self.N, self.H).permute(0, 2, 1, 3)
        k = k.contiguous().reshape(B, S, self.N, self.H).permute(0, 2, 1, 3)
        v = v.contiguous().reshape(B, S, self.N, self.H).permute(0, 2, 1, 3)

        if self.shared_k:
            k = k[:, :1] # simply ignore the 2nd head's k

        attn_weight = (q @ k.transpose(-1, -2)) * self.qk_scale # (B, N, T, S)
        
        if attn_mask is not None: 
            if attn_mask.dim() == 3: # (B, T, S) instance-specific 
                attn_mask = attn_mask.unsqueeze(1)
            elif attn_mask.dim() == 2: #  (T, S) shared within the batch
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_weight.masked_fill_(attn_mask, float('-inf'))

        if key_padding_mask is not None: # (B, T) instance-specific
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weight.masked_fill_(attn_mask, float('-inf'))

        #attn_weight = self.attn_dp(attn_weight.softmax(dim=-1))
        #x = (attn_weight @ v).transpose(1, 2).reshape(B, T, self.D)
        #x = self.proj_dp(self.proj(x))

        indice1 = torch.arange(S, device=v.device).repeat_interleave(S)
        indice2 = torch.arange(S, device=v.device).repeat(S)

        if self.mode == 0: # two versions of v and two versions of dist.
            v1 = v[:, 0, indice1]
            v2 = v[:, 1, indice2]
            x = torch.cat([v1, v2], dim=-1)

            if not self.routing:
                pass #x = self.proj_dp(self.proj(x))

            p = attn_weight.log_softmax(dim=-1)

            p1 = p[:, 0, :, indice1]
            p2 = p[:, 1, :, indice2]
            pair_logit = p = p1 + p2

        elif self.mode == 1: # one version of v and two versions of dist.
            v = v.permute(0, 2, 1, 3).reshape(B, S, -1)
            v1 = v[:, indice1]
            v2 = v[:, indice2]
            x = torch.cat([v1, v2], dim=-1)

            if not self.routing:
                x = self.proj_dp(self.proj(x))

            p = attn_weight.log_softmax(dim=-1)

            p1 = p[:, 0, :, indice1]
            p2 = p[:, 1, :, indice2]
            pair_logit = p = p1 + p2

        elif self.mode == 2: # one version of v and two versions of dist.
            attn_weight = self.attn_dp(attn_weight.softmax(dim=-1))
            v = v.permute(0, 2, 1, 3).reshape(B, S, -1)
            if self.training:
                v = v.unsqueeze(1)
                x = (attn_weight @ v).transpose(1, 2).reshape(B, T, -1)
            else:
                indice = attn_weight.argmax(-1).transpose(1, 2).reshape(B, -1).unsqueeze(-1).expand(-1, -1, v.shape[-1])
                x = v.gather(1, indice).reshape(B, T, -1)

            if not self.routing:
                x = self.proj_dp(self.proj(x))

            pair_logit = torch.zeros((B, T, 1), device=x.device)

        if self.routing:
            attn_weight_ = self.attn_dp(pair_logit.softmax(dim=-1))
            x = attn_weight_ @ x
            x = self.proj_dp(self.proj(x))
            return x, attn_weight

        return x, (attn_weight, pair_logit)
