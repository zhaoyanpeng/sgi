import torch
import torch.nn as nn
import torch.nn.functional as F

from sgi.module import sparsemax

# This class is mainly used by decoder.py for RNNs but also
# by the CNN / transformer decoder when copy attention is used
# CNN has its own attention mechanism ConvMultiStepAttention
# Transformer has its own MultiHeadedAttention

from fvcore.common.registry import Registry

ATTENTION_HEADS_REGISTRY = Registry("ATTENTION_HEADS")
ATTENTION_HEADS_REGISTRY.__doc__ = """
Registry for encoder heads.
"""

def build_attention_head(cfg, **kwargs):
    return ATTENTION_HEADS_REGISTRY.get(cfg.name)(cfg, **kwargs)


@ATTENTION_HEADS_REGISTRY.register()
class GlobalAttention(nn.Module):
    r"""
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.
    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.
    .. mermaid::
       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G
    All models compute the output as
    :math:`c = \sum_{j=1}^{\text{SeqLength}} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].
    However they
    differ on how they compute the attention score.
    * Luong Attention (dot, general):
       * dot: :math:`\text{score}(H_j,q) = H_j^T q`
       * general: :math:`\text{score}(H_j, q) = H_j^T W_a q`
    * Bahdanau Attention (mlp):
       * :math:`\text{score}(H_j, q) = v_a^T \text{tanh}(W_a q + U_a h_j)`
    Args:
       dim (int): dimensionality of query and key
       attn_type (str): type of attention to use, options [dot,general,mlp]
       attn_func (str): attention function to use, options [softmax,sparsemax]
    """

    def __init__(self, cfg, src_dim=None, tgt_dim=None, **kwargs):
        super(GlobalAttention, self).__init__()
        src_dim = src_dim or cfg.src_dim
        tgt_dim = tgt_dim or cfg.tgt_dim
        attn_type = cfg.attn_type
        attn_func = cfg.attn_func

        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.dim = cfg.attn_size

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

    def forward(self, source, memory, memory_mask=None, **kwargs):
        """
        Args:
          source (FloatTensor): query vectors ``(batch, tgt_len, dim)``
          memory (FloatTensor): source vectors ``(batch, src_len, dim)``
          memory_mask (LongTensor): the source context masks ``(batch, src_len)``
        Returns:
          (FloatTensor, FloatTensor, FloatTensor):
          * Computed vector ``(tgt_len, batch, dim)``
          * Attention distribtutions ``(tgt_len, batch, src_len)``
          * Contexts ``(tgt_len, batch, src_len)``
        """
        # one step input
        one_step = False
        if source.dim() == 2:
            source = source.unsqueeze(1)
            one_step = True

        B, S, H_s = memory.size()
        B, T, H_t = source.size()

        align = self.score(source, memory)

        if memory_mask is not None:
            assert memory_mask.dim() == 2
            mask = memory_mask.unsqueeze(1)
            align.masked_fill_(mask, float('-inf'))

        attention = self.attn_func(align.view(B * T, S), -1)
        attention = attention.view(B, T, S)

        c = torch.bmm(attention, memory)

        ctx_source = torch.cat([c, source], 2).view(B * T, H_s + H_t)
        attn_h = self.linear_out(ctx_source).view(B, T, H_t)
        attn_h = self.activation(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            attention = attention.squeeze(1)
            c = c.squeeze(1)
        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            attention = attention.transpose(0, 1).contiguous()
            c = c.transpose(0, 1).contiguous()

        return attn_h, attention, c
