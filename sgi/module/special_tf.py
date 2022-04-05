from typing import Callable, Dict, List, Optional, Tuple, Union
import torch.nn.functional as F
from torch import nn, Tensor
import torch

from . import MetaModule, MiniTFBlock, MiniTFAttention

__all__ = [
    "SpecialTFBlock",
    "SpecialTFAttention",
]


class SpecialTFBlock(MiniTFBlock):
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
            x = x.unsqueeze(1)
            residual = residual.unsqueeze(2)
            x = self.inter_attn_ln(residual + self.inter_attn_dp(x))

        x = self.ff_ln(x + self.ff_dp(self.ff(x)))
        return x, (intra_attn_weight, inter_attn_weight)


class SpecialTFAttention(MiniTFAttention):
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

        v1 = v[:, 0, indice1]
        v2 = v[:, 1, indice2]
        x = torch.cat([v1, v2], dim=-1)
        x = self.proj_dp(self.proj(x))
        
        p = attn_weight.log_softmax(dim=-1)

        p1 = p[:, 0, :, indice1]
        p2 = p[:, 1, :, indice2]
        pair_logit = p = p1 + p2

        return x, (attn_weight, pair_logit)

