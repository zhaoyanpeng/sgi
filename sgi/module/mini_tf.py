from typing import Callable, Dict, List, Optional, Tuple, Union
import torch.nn.functional as F
from torch import nn, Tensor
import torch

from . import sign

__all__ = [
    "MiniTF", 
    "MiniTFBlock",
    "SignTFBlock",
    "SortTFAttention",
    "MiniTFAttention",
    "SignTFAttention",
    "FakeTFAttention",
    "RelationTFAttention",
    "_get_activation_fn",
    "_get_initializr_fn",
    "_get_clones",
]

_get_activation_fn = {
    "elu": nn.ELU(),
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "celu": nn.CELU(),
    "tanh": nn.Tanh(),
    "lelu": nn.LeakyReLU(),
    "none": nn.Identity(),
}

_get_initializr_fn = {
    "norm": nn.init.normal_,
    "xavier": nn.init.xavier_uniform_,
}

def _get_clones(module_fn, N):
    return nn.ModuleList([module_fn(i) for i in range(N)])

class MetaModule(nn.Module):
    """ A nicer __repr__.
    """
    def __init__(self):
        super().__init__()

    def extra_repr(self):
        mod_keys = self._modules.keys()
        all_keys = self._parameters.keys()
        extra_keys = all_keys - mod_keys
        extra_keys = [k for k in all_keys if k in extra_keys]
        extra_lines = []
        for key in extra_keys:
            attr = getattr(self, key)
            if not isinstance(attr, nn.Parameter):
                continue
            extra_lines.append("({}): Tensor{}".format(key, tuple(attr.size())))
        return "\n".join(extra_lines)

class MiniTF(nn.Module):
    """ A shell for both encoder and decoder.
    """
    def __init__(self, layer: nn.Module, N: int, layer_norm: Callable = lambda x: x):
        super().__init__()
        self.layers = _get_clones(layer, N) 
        self.layer_norm = layer_norm 

    def forward(
        self, x: Tensor, 
        kv: Tensor = None, 
        self_attn_mask: Tensor = None, 
        self_key_padding_mask: Tensor = None,
        memory: Tensor = None,
        memo_attn_mask: Tensor = None,
        memo_key_padding_mask: Tensor = None,
        require_attn_weight: bool = False,
        **kwargs
    ): 
        attn_weights = list()
        for ilayer, layer in enumerate(self.layers):
            x, attn_weight = layer(
                x, kv = kv, 
                self_attn_mask = self_attn_mask, 
                self_key_padding_mask = self_key_padding_mask,
                memory = memory, 
                memo_attn_mask = memo_attn_mask, 
                memo_key_padding_mask = memo_key_padding_mask,
                **kwargs
            )
            attn_weights.append(attn_weight)
        x = self.layer_norm(x) # x has already been normed at the end of each layer
        return x, (attn_weights if require_attn_weight else None)

class MiniTFBlock(MetaModule):
    """ Encoder or decoder, it is your choice.
    """
    def __init__(
        self, D: int, N: int, F: int, 
        attn_cls_intra, 
        attn_cls_inter: str = None,
        ilayer: int = 0,
        dropout: float = .0, 
        qk_scale: float = None,
        activation: str = "gelu",
        attn_dropout: float = .0,
        proj_dropout: float = .0,
        num_head_intra: int = None,
        num_head_inter: int = None,
        q_activation: str = "none",
        k_activation: str = "none",
        sign_q_intra: bool = False,
        sign_k_intra: bool = False,
        sign_q_inter: bool = False,
        sign_k_inter: bool = False,
        inter_layers: list = [],
        **kwargs,
    ):
        super().__init__()
        self.intra_attn = eval(attn_cls_intra)(
            D, num_head_intra or N, attn_dropout=attn_dropout, proj_dropout=proj_dropout,
            sign_q=sign_q_intra, sign_k=sign_k_intra, q_activation=q_activation, k_activation=k_activation, **kwargs
        ) 
        self.intra_attn_ln = nn.LayerNorm(D)
        self.intra_attn_dp = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(D, F),
            _get_activation_fn.get(activation, nn.GELU),
            nn.Dropout(dropout),
            nn.Linear(F, D), 
        )
        self.ff_ln = nn.LayerNorm(D) 
        self.ff_dp = nn.Dropout(dropout)

        do_inter = True if ilayer >= len(inter_layers) else inter_layers[ilayer]

        if do_inter and attn_cls_inter is not None:
            self.inter_attn = eval(attn_cls_inter)(
                D, num_head_inter or N, attn_dropout=attn_dropout, proj_dropout=proj_dropout,
                sign_q=sign_q_inter, sign_k=sign_k_inter, q_activation=q_activation, k_activation=k_activation, **kwargs
            )
            self.inter_attn_ln = nn.LayerNorm(D)
            self.inter_attn_dp = nn.Dropout(dropout) 
        else:
            self.register_parameter("inter_attn", None)
            self.register_parameter("inter_attn_ln", None)
            self.register_parameter("inter_attn_dp", None)
        self._reset_parameters()

    def _reset_parameters(self):
        pass

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
            x = self.inter_attn_ln(residual + self.inter_attn_dp(x))

        x = self.ff_ln(x + self.ff_dp(self.ff(x)))
        return x, (intra_attn_weight, inter_attn_weight)

class MiniTFAttention(MetaModule):
    """ Light-weight MHA for batch-first inputs.
    """
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
        **kwargs,
    ):
        super().__init__()
        assert D % N == 0
        self.D = D   
        self.N = N
        self.H = D // N 

        self.qk_scale = qk_scale or self.H ** -0.5
        
        self.kdim = kdim if kdim is not None else D
        self.vdim = vdim if vdim is not None else D

        if self.kdim == self.vdim == D:
            self.proj_weight = nn.Parameter(Tensor(3 * D, D))
            self.register_parameter("q_proj", None)
            self.register_parameter("k_proj", None)
            self.register_parameter("v_proj", None)
            num_bias = 3 * D
        elif self.kdim == self.vdim:
            self.proj_weight = nn.Parameter(Tensor(2 * D, kdim))
            self.q_proj = nn.Linear(D, 1 * D, bias=bias)
            self.register_parameter("k_proj", None)
            self.register_parameter("v_proj", None)
            num_bias = 2 * D
        else:
            self.register_parameter("proj_weight", None)
            self.q_proj = nn.Linear(D, 1 * D, bias=bias)
            self.k_proj = nn.Linear(kdim, 1 * D, bias=bias)
            self.v_proj = nn.Linear(vdim, 1 * D, bias=bias)
            num_bias = 0 * D

        if bias and num_bias > 0:
            self.proj_bias = nn.Parameter(Tensor(num_bias)) 
        else:
            self.register_parameter("proj_bias", None)

        self.proj = nn.Linear(D, D)
        self.proj_dp = nn.Dropout(proj_dropout)
        self.attn_dp = nn.Dropout(attn_dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        if self.proj_weight is not None:
            nn.init.xavier_uniform_(self.proj_weight) 
        if self.proj_bias is not None:
            nn.init.constant_(self.proj_bias, 0.)

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

        attn_weight = self.attn_dp(attn_weight.softmax(dim=-1))
        x = (attn_weight @ v).transpose(1, 2).reshape(B, T, self.D)
        x = self.proj_dp(self.proj(x))
        return x, attn_weight

    def _proj_qkv(self, x):
        return self._in_proj(x).chunk(3, dim=-1)

    def _proj_kv(self, x):
        return self._in_proj(x, end=2 * self.D).chunk(2, dim=-1)

    def _proj_k(self, x):
        return (self._in_proj(x, end=self.D) 
            if self.k_proj is None else self.k_proj(x)
        )
    def _proj_v(self, x):
        return (self._in_proj(x, start=self.D, end=2 * self.D) 
            if self.v_proj is None else self.v_proj(x)
        )
    def _proj_q(self, x):
        return (self._in_proj(x, start=2 * self.D) 
            if self.q_proj is None else self.q_proj(x)
        )
    def _in_proj(self, x, start=0, end=None):
        weight = self.proj_weight[start : end]
        bias = (
            None if self.proj_bias is None else self.proj_bias[start : end]
        )
        return F.linear(x, weight, bias)

class RelationTFAttention(MiniTFAttention):
    def forward(
        self, 
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor = None,
        key_padding_mask: Tensor = None,
        relation_k: Tensor = None,
        relation_v: Tensor = None,
        **kwargs
    ):
        r_k, r_v = relation_k, relation_v # (B, N, T, S, H)

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

        # relation k
        attn_weight_rel = (r_k @ q.unsqueeze(-1)).squeeze(-1) # (B, N, T, H, 1) 
        attn_weight = attn_weight + attn_weight_rel 
        
        if attn_mask is not None: 
            if attn_mask.dim() == 3: # (B, T, S) instance-specific 
                attn_mask = attn_mask.unsqueeze(1)
            elif attn_mask.dim() == 2: #  (T, S) shared within the batch
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_weight.masked_fill_(attn_mask, float('-inf'))

        if key_padding_mask is not None: # (B, T) instance-specific
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weight.masked_fill_(attn_mask, float('-inf'))

        # (B, N, T, S) x (B, N, S, H) -> (B, N, T, H)
        attn_weight = self.attn_dp(attn_weight.softmax(dim=-1))
        x = (attn_weight @ v).transpose(1, 2).reshape(B, T, self.D)

        # relation v # (B, N, T, 1, S) 
        x_rel = (attn_weight.unsqueeze(-2) @ r_v).transpose(1, 2).reshape(B, T, self.D)
        x = x + x_rel

        x = self.proj_dp(self.proj(x))
        return x, attn_weight

class SignTFAttention(MiniTFAttention):
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
        q_activation: str = "none",
        k_activation: str = "none",
        sign_q: bool = False,
        sign_k: bool = False,
        **kwargs,
    ):
        super().__init__(
            D, N,
            kdim=kdim,
            vdim=vdim,
            bias=bias,
            qk_scale=qk_scale,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.sign_q = sign_q
        self.sign_k = sign_k
        if self.kdim == D:
            nchunk = self.sign_q + self.sign_k
            self.sign_proj = nn.Linear(D, nchunk * D, bias=bias)
            self.register_parameter("q_sign_proj", None)
            self.register_parameter("k_sign_proj", None)
        else:
            self.register_parameter("sign_proj", None)
            if not self.sign_q:
                self.register_parameter("q_sign_proj", None)
            else:
                self.q_sign_proj = nn.Linear(D, 1 * D, bias=bias)
            if not self.sign_k:
                self.register_parameter("k_sign_proj", None)
            else:
                self.k_sign_proj = nn.Linear(kdim, 1 * D, bias=bias)
        self.q_activation = _get_activation_fn.get(q_activation, nn.Identity())
        self.k_activation = _get_activation_fn.get(k_activation, nn.Identity())

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor = None,
        key_padding_mask: Tensor = None,
        **kwargs
    ):
        old_q, old_k = q, k
        if q.data_ptr() == k.data_ptr() == v.data_ptr():
            k, v, q = self._proj_qkv(q)
        elif k.data_ptr() == v.data_ptr():
            k, v = self._proj_kv(k)
            q = self._proj_q(q)
        else:
            q = self._proj_q(q)
            k = self._proj_k(k)
            v = self._proj_v(v)

        if old_q.data_ptr() == old_k.data_ptr():
            sq, sk = self._sign_proj_qk(old_q)
        else:
            sq = self._sign_proj_q(old_q)
            sk = self._sign_proj_k(old_k)
        sq = q if sq is None else sq
        sk = k if sk is None else sk

        B, T, S = q.shape[0], q.shape[1], k.shape[1]

        # (B, L, D) -> (B, L, N, H) -> (B, N, L, H)
        q = q.contiguous().reshape(B, T, self.N, self.H).permute(0, 2, 1, 3)
        k = k.contiguous().reshape(B, S, self.N, self.H).permute(0, 2, 1, 3)
        v = v.contiguous().reshape(B, S, self.N, self.H).permute(0, 2, 1, 3)

        attn_weight = (q @ k.transpose(-1, -2)) * self.qk_scale # (B, N, T, S)

        sq = sq.contiguous().reshape(B, T, self.N, self.H).permute(0, 2, 1, 3)
        sk = sk.contiguous().reshape(B, S, self.N, self.H).permute(0, 2, 1, 3)
        
        sq = self.q_activation(sq)
        sk = self.k_activation(sk)

        sign_weight = sq @ sk.transpose(-1, -2) # (B, N, T, S)
        sign_weight = sign_weight.tanh() # nn.Softsign()(sign_weight) #

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
        #attn_weight = attn_weight * sign_weight

        sign_weight = sign_weight.masked_fill(attn_weight == float('-inf'), 0.)
        ##sign_scale = sign_weight.sum(-1, keepdim=True).abs() # zero alert
        attn_weight = sign_weight

        x = (attn_weight @ v).transpose(1, 2).reshape(B, T, self.D)
        x = self.proj_dp(self.proj(x))
        return x, attn_weight

    def _sign_proj_qk(self, x):
        #qk = self._sign_in_proj(x)
        qk = self.sign_proj(x)
        if qk.shape[-1] == 0:
            q, k = None, None
        elif qk.shape[-1] == 2 * self.D:
            q, k = qk.chunk(2, dim=-1)
        elif self.sign_q:
            q, k = qk, None
        elif self.sign_k:
            q, k = None, qk
        return q, k

    def _sign_proj_q(self, x):
        return ((self._sign_in_proj(x, end=self.D)
            if self.q_sign_proj is None else self.q_sign_proj(x)
        ) if self.sign_q else None)

    def _sign_proj_k(self, x):
        return ((self._sign_in_proj(x, start=self.D)
            if self.k_sign_proj is None else self.k_sign_proj(x)
        ) if self.sign_k else None)

    def _sign_in_proj(self, x, start=0, end=None):
        if self.sign_proj.out_features == start:
            end = start
            start = 0
        weight = self.sign_proj.weight[start : end]
        bias = (
            None if self.sign_proj.bias is None else self.sign_proj.bias[start : end]
        )
        return F.linear(x, weight, bias)

class SortTFAttention(SignTFAttention):
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
        q_activation: str = "none",
        k_activation: str = "none",
        sign_q: bool = False,
        sign_k: bool = False,
        num_pos: int = 10,
        **kwargs,
    ):
        super().__init__(
            D, N,
            kdim=kdim,
            vdim=vdim,
            bias=bias,
            qk_scale=qk_scale,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            q_activation=q_activation,
            k_activation=k_activation,
            sign_q=sign_q,
            sign_k=sign_k,
        )
        # may need to interpolate pos for longer sequences
        self.proj_pos = nn.Parameter(Tensor(num_pos, self.D, self.D))
        nn.init.xavier_uniform_(self.proj_pos)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor = None,
        key_padding_mask: Tensor = None,
        **kwargs
    ):
        old_q, old_k, old_v = q, k, v
        if q.data_ptr() == k.data_ptr() == v.data_ptr():
            k, v, q = self._proj_qkv(q)

            # currently only works for intra attention
            # it does not make sense to use it in inter attention

            B, L, D = v.shape[:3]
            new_p = self.proj_pos[:L]
            pv = old_v.unsqueeze(-2) @ new_p.unsqueeze(0) # (B, L, D)
            pv = pv.squeeze(-2)

            v = pv # + v

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


        """
        #masking out `self' could result in faster convergence but is not necessary
        self_mask = torch.full((T, S), 0, device=q.device).bool()
        indice = torch.arange(T, device=q.device)
        self_mask[indice, indice] = True

        self_mask = self_mask.unsqueeze(0).unsqueeze(1)
        attn_weight.masked_fill_(self_mask, float("-inf"))
        """

        attn_weight = self.attn_dp(attn_weight.softmax(dim=-1))
        x = (attn_weight @ v).transpose(1, 2).reshape(B, T, self.D)
        x = self.proj_dp(self.proj(x))
        return x, attn_weight




class SignTFBlock(MetaModule):
    """ Encoder or decoder, it is your choice.
    """
    def __init__(
        self, D: int, N: int, F: int,
        attn_cls_intra,
        attn_cls_inter: str = None,
        ilayer: int = 0,
        dropout: float = .0,
        qk_scale: float = None,
        activation: str = "gelu",
        attn_dropout: float = .0,
        proj_dropout: float = .0,
        num_head_intra: int = None,
        num_head_inter: int = None,
        q_activation: str = "none",
        k_activation: str = "none",
        sign_q_intra: bool = False,
        sign_k_intra: bool = False,
        sign_q_inter: bool = False,
        sign_k_inter: bool = False,
        inter_layers: list = [],
        **kwargs,
    ):
        super().__init__()
        ln_output = True
        self.intra_attn = eval(attn_cls_intra)(
            D, num_head_intra or N, attn_dropout=attn_dropout, proj_dropout=proj_dropout,
            sign_q=sign_q_intra, sign_k=sign_k_intra, q_activation=q_activation, k_activation=k_activation, **kwargs
        )
        self.intra_attn_ln = nn.LayerNorm(D) if ln_output else nn.Identity()
        self.intra_attn_dp = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(D, F),
            _get_activation_fn.get(activation, nn.GELU()),
            nn.Dropout(dropout),
            nn.Linear(F, D),
        )
        self.ff_ln = nn.LayerNorm(D) if ln_output else nn.Identity()
        self.ff_dp = nn.Dropout(dropout)

        do_inter = True if ilayer >= len(inter_layers) else inter_layers[ilayer]

        if do_inter and attn_cls_inter is not None:
            self.inter_attn = eval(attn_cls_inter)(
                D, num_head_inter or N, attn_dropout=attn_dropout, proj_dropout=proj_dropout,
                sign_q=sign_q_inter, sign_k=sign_k_inter, q_activation=q_activation, k_activation=k_activation, **kwargs
            )
            self.inter_attn_ln = nn.LayerNorm(D) if ln_output else nn.Identity()
            self.inter_attn_dp = nn.Dropout(dropout)
        else:
            self.register_parameter("inter_attn", None)
            self.register_parameter("inter_attn_ln", None)
            self.register_parameter("inter_attn_dp", None)
        self._reset_parameters()

    def _reset_parameters(self):
        pass

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
            x = self.inter_attn_ln(residual + self.inter_attn_dp(x))

        x = self.ff_ln(x + self.ff_dp(self.ff(x)))
        return x, (intra_attn_weight, inter_attn_weight)

class FakeTFAttention(SignTFAttention):
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
        q_activation: str = "none",
        k_activation: str = "none",
        sign_q: bool = False,
        sign_k: bool = False,
        **kwargs,
    ):
        super().__init__(
            D, N,
            kdim=kdim,
            vdim=vdim,
            bias=bias,
            qk_scale=qk_scale,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            q_activation=q_activation,
            k_activation=k_activation,
            sign_q=sign_q,
            sign_k=sign_k,
        )
        # concat the contexts and do linear transformation
        self.case = 2 
        if self.case in {1}:
            self.linear = nn.Linear(self.D * 2, D, bias=False) 
        elif self.case in {2}:
            L = 8 
            self.positional_linear = nn.Parameter(Tensor(L, self.D, self.D))

    def forward(
        self, 
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor = None,
        key_padding_mask: Tensor = None,
        **kwargs
    ):
        if self.case == 1:

            B, L, H = q.shape
            
            s = q[:, 0:1]
            x = q[:, 1:2]
            y = q[:, 2:3]
            z = q[:, 3:4]
            e = q[:, 4:5]

            xx = torch.cat([y, z], dim=-1)
            yy = torch.cat([x, z], dim=-1)
            zz = torch.cat([x, y], dim=-1)

            xyz = torch.cat([xx, yy, zz], dim=1)
            xyz = self.linear(xyz)

            q = h = torch.cat([s, xyz, e], dim=1)

            return q, None

        elif self.case == 2:

            old_q, old_k, old_v = q, k, v
            if q.data_ptr() == k.data_ptr() == v.data_ptr():
                k, v, q = self._proj_qkv(q)

                B, L, D = v.shape[:3]
                new_p = self.positional_linear[:L]
                pv = old_v.unsqueeze(-2) @ new_p.unsqueeze(0) # (B, L, D) 
                pv = pv.squeeze(-2)

                v = pv # + v

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


            """ could converge faster but not necessary
            self_mask = torch.full((T, S), 0, device=q.device).bool()
            indice = torch.arange(T, device=q.device)
            self_mask[indice, indice] = True

            self_mask = self_mask.unsqueeze(0).unsqueeze(1)
            attn_weight.masked_fill_(self_mask, float("-inf"))
            """
            #print(attn_weight)


            attn_weight = self.attn_dp(attn_weight.softmax(dim=-1))
            x = (attn_weight @ v).transpose(1, 2).reshape(B, T, self.D)
            x = self.proj_dp(self.proj(x))
            return x, attn_weight

        elif self.case == 3:
            old_q, old_k = q, k
            if q.data_ptr() == k.data_ptr() == v.data_ptr():
                k, v, q = self._proj_qkv(q)
            elif k.data_ptr() == v.data_ptr():
                k, v = self._proj_kv(k)
                q = self._proj_q(q)
            else:
                q = self._proj_q(q)
                k = self._proj_k(k)
                v = self._proj_v(v)

            if old_q.data_ptr() == old_k.data_ptr():
                sq, sk = self._sign_proj_qk(old_q)
            else:
                sq = self._sign_proj_q(old_q)
                sk = self._sign_proj_k(old_k)
            sq = q if sq is None else sq
            sk = k if sk is None else sk

            B, T, S = q.shape[0], q.shape[1], k.shape[1]

            # (B, L, D) -> (B, L, N, H) -> (B, N, L, H)
            q = q.contiguous().reshape(B, T, self.N, self.H).permute(0, 2, 1, 3)
            k = k.contiguous().reshape(B, S, self.N, self.H).permute(0, 2, 1, 3)
            v = v.contiguous().reshape(B, S, self.N, self.H).permute(0, 2, 1, 3)

            attn_weight = (q @ k.transpose(-1, -2)) * self.qk_scale # (B, N, T, S)

            sq = sq.contiguous().reshape(B, T, self.N, self.H).permute(0, 2, 1, 3)
            sk = sk.contiguous().reshape(B, S, self.N, self.H).permute(0, 2, 1, 3)
            
            sq = self.q_activation(sq)
            sk = self.k_activation(sk)

            sign_weight = sq @ sk.transpose(-1, -2) # (B, N, T, S)
            sign_weight = sign_weight.tanh() # or nn.Softsign()

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
            #attn_weight = attn_weight * sign_weight

            sign_weight = sign_weight.masked_fill(attn_weight == float('-inf'), 0.)
            ##sign_scale = sign_weight.sum(-1, keepdim=True).abs() # zero alert
            attn_weight = sign_weight

            x = (attn_weight @ v).transpose(1, 2).reshape(B, T, self.D)
            x = self.proj_dp(self.proj(x))
            return x, attn_weight
