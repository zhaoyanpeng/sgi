import numpy as np
import os, sys, time
import torch
from typing import Tuple
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import torch.nn.functional as F

from fvcore.common.registry import Registry

from .. import PositionalEncoder, PartiallyFixedEmbedding
from .. import MiniTF, MiniTFBlock, MiniTFAttention, RelationTFAttention

ENCODER_HEADS_REGISTRY = Registry("ENCODER_HEADS")
ENCODER_HEADS_REGISTRY.__doc__ = """
Registry for encoder heads.
"""

initializr = lambda x: None 

def build_encoder_head(cfg, vocab, **kwargs):
    return ENCODER_HEADS_REGISTRY.get(cfg.name)(cfg, vocab, **kwargs)

@ENCODER_HEADS_REGISTRY.register()
class DummyEncHead(nn.Module):
    def __init__(self, cfg, token_vocab, **kwargs):
        super().__init__()
        pass
    def _reset_parameters(self):
        pass
    def output_shape(self):
        return 0 
    def forward(self, *args, **kwargs):
        return None, None 

class MetaEncHead(nn.Module):
    """ TODO: This should be a standalone backbone module.
    """
    def __init__(self, cfg, token_vocab):
        super().__init__()
        # input embedding layer
        wdim = cfg.w_dim
        self.token_embed = None
        self.token_vocab = token_vocab
        if os.path.isfile(cfg.w2v_file):
            self.token_embed = PartiallyFixedEmbedding(self.token_vocab, cfg.w2v_file)
        elif not cfg.skip_embed:
            if not cfg.w_onehot:
                self.token_embed = nn.Embedding(
                    len(self.token_vocab), wdim, padding_idx=self.token_vocab.PAD_IDX
                )
            else:
                nword = len(self.token_vocab)
                real_wdim = wdim // nword # each onehot sub-vector indicates a word
                assert real_wdim > 0, f"word dim ({real_wdim}) shoule be above 0." 
                weight = torch.zeros(nword, wdim)
                for i in range(k):
                    if i == self.token_vocab.PAD_IDX:
                        continue
                    weight[i, i * real_wdim : (i + 1) * real_wdim] = 1.
                self.token_embed = nn.Embedding.from_pretrained(
                    weight, freeze=True, padding_idx=self.token_vocab.PAD_IDX
                )
        # input positional embedding 
        input_channels = wdim * cfg.num_w if cfg.cat_w else wdim
        num_p, p_dim, cat_p, p_type = cfg.num_p, cfg.p_dim, cfg.cat_p, cfg.p_type
        if p_dim <= 0 or num_p <= 0:
            self.register_parameter("position_embed", None)
            if cat_p:
                input_channels += 4
        elif p_type == "sinuous":
            self.position_embed = PositionalEncoder(p_dim, dropout=cfg.p_dropout)
            if cat_p:
                input_channels += p_dim * 4
        elif p_type == "learned":
            self.position_embed = nn.Linear(num_p, p_dim, bias=False)
            if cat_p:
                input_channels += p_dim * 1 

        self.input_channels = input_channels
        if self.input_channels == cfg.m_dim: 
            self.register_parameter("fc0", None)
        else:
            self.fc0 = nn.Linear(self.input_channels, cfg.m_dim, bias=False) 
        self.ln0 = nn.LayerNorm(cfg.m_dim) if cfg.ln_input else nn.Identity()
        
        self._output_size = self._emb_size = cfg.m_dim

        self.p_dim = p_dim
        self.cat_p = cat_p

    @property
    def emb_size(self):
        return self._emb_size

    @property
    def output_size(self):
        return self._output_size

    def _reset_parameters(self):
        for field in [self.fc0]:
            if field is None: continue
            if hasattr(field, "weight") and field.weight is not None:
                initializr(field.weight) 
            if hasattr(field, "bias") and field.bias is not None:
                nn.init.constant_(field.bias, 0.)

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

    def _encode_positions(self, x, bbox, img_shape):
        if self.position_embed is not None:
            #assert bbox is not None
            if bbox is None:
                positions = self.position_embed.weight[:, :x.shape[1]]
                positions = positions.transpose(0, 1).unsqueeze(0).expand(x.shape[0], -1, -1)
            elif bbox.dtype == torch.int64: # integer indice
                shape = bbox.shape + (-1,)
                positions = self.position_embed.weight[:, bbox.reshape(-1)] # bias discarded
                positions = positions.transpose(0, 1).view(shape)
            elif isinstance(self.position_embed, PositionalEncoder):
                (h, w) = img_shape[:2]
                bbox = bbox.clone()
                bbox[:, :, 0::2] *= w # w 
                bbox[:, :, 1::2] *= h # h 
                shape = bbox.shape[:2] + (-1,)
                positions = self.position_embed.encode(bbox.long())
                positions = positions.view(shape)
            else:
                bbox = bbox.clone()
                #bbox[:, :, 1:] = 0. # only x matters for left / right relation
                #bbox = bbox * 10 # might make it easier to learn?
                positions = self.position_embed(bbox)
                #positions = F.relu(positions)
            if self.cat_p:
                x = torch.cat([x, positions], -1)
            else:
                assert list(x.shape) == [] or x.shape == positions.shape
                x = x + positions
        else: # cfg.p_dim <= 0 or cfg.num_p <= 0
            if self.cat_p: # manipulate the original positions
                assert bbox is not None
                if self.p_dim == 4 or self.p_dim == 128:
                    x = torch.cat([x, bbox], -1)
                elif self.p_dim == 2:
                    cx = (bbox[:, :, [0]] + bbox[:, :, [2]]) / 2
                    cy = (bbox[:, :, [1]] + bbox[:, :, [3]]) / 2
                    pad = torch.zeros_like(cx)
                    x = torch.cat([x, cx, cy, pad, pad], -1)
                elif self.p_dim == 1:
                    #cx = bbox[:, :, [0]] # x axis
                    cx = (bbox[:, :, [0]] + bbox[:, :, [2]]) / 2
                    pad = torch.zeros_like(cx)
                    x = torch.cat([x, cx, pad, pad, pad], -1)
        x = self.fc0(x) if self.fc0 is not None else x
        x = self.ln0(x)
        return x

@ENCODER_HEADS_REGISTRY.register()
class MiniTFEncHead(MetaEncHead):
    """ Customized Transformer encoder.
    """
    def __init__(self, cfg, token_vocab, **kwargs):
        super().__init__(cfg, token_vocab)
        self.encoder = None
        if cfg.num_layer > 0: 
            layer_fn = lambda ilayer: MiniTFBlock(
                cfg.m_dim, cfg.num_head, cfg.f_dim, cfg.attn_cls_intra,
                attn_cls_inter=cfg.attn_cls_inter,
                ilayer=ilayer,
                dropout=cfg.t_dropout,
                qk_scale=cfg.qk_scale,
                activation=cfg.activation,
                attn_dropout=cfg.attn_dropout,
                proj_dropout=cfg.proj_dropout,
                num_head_intra=cfg.num_head_intra,
                num_head_inter=cfg.num_head_inter,
                q_activation=cfg.q_activation,
                k_activation=cfg.k_activation,
                sign_q_intra=cfg.sign_q,
                sign_k_intra=cfg.sign_k,
                num_pos=cfg.max_enc_len,
            )
            self.encoder = MiniTF(layer_fn, cfg.num_layer)

        self._reset_parameters()

    def forward(
        self, 
        x: Tensor, *args,
        bbox: Tensor=None,
        img_shape: Tuple=(320, 480),
        self_attn_mask: Tensor=None,
        self_key_padding_mask: Tensor=None,
        attn_weight_type: str=None,
        **kwargs
    ):
        #bbox = None if True else bbox
        x = self._encode_positions(x, bbox, img_shape)

        """
        y = x[:, :, 256:260]
        print(y)
        b = bbox[:, :, :1]
        print(b)
        e = self.position_embed.weight[:4, 0]
        print(e)
        f = y / e.unsqueeze(0).unsqueeze(0)
        print(f)
        """

        if self.encoder is None:
            return x, None, None

        x, _ = self.encoder(
            x, self_key_padding_mask=self_key_padding_mask, **kwargs,
        ) 

        return x, None, None

@ENCODER_HEADS_REGISTRY.register()
class TorchTFEncHead(MetaEncHead):
    """ Standard Transformer encoder.
    """
    def __init__(self, cfg, token_vocab, **kwargs):
        super().__init__(cfg, token_vocab)
        self.encoder = None
        if cfg.num_layer > 0:
            layer_fn = TransformerEncoderLayer(
                cfg.m_dim, cfg.num_head, cfg.f_dim, cfg.t_dropout, activation=cfg.activation,
            )
            self.encoder = TransformerEncoder(layer_fn, cfg.num_layer)

        self._reset_parameters()

    def forward(
        self, 
        x: Tensor, *args,
        bbox: Tensor=None,
        img_shape: Tuple=(320, 480),
        self_attn_mask: Tensor=None,
        self_key_padding_mask: Tensor=None,
        attn_weight_type: str=None,
        **kwargs
    ):
        x = self._encode_positions(x, bbox, img_shape)

        if self.encoder is None:
            return x, None, None
        
        x = x.transpose(0, 1)

        x = self.encoder(
            x, src_key_padding_mask=self_key_padding_mask,
        ) 

        x = x.transpose(0, 1)

        return x, None, None

@ENCODER_HEADS_REGISTRY.register()
class RelationDistiller(MiniTFEncHead):
    """ Relation Induction: distilling relations into multihead relations.
    """
    def __init__(self, cfg, token_vocab, **kwargs):
        super().__init__(cfg, token_vocab)
        # FAST tri-linear layer for relation distribution prediction
        self.proj_fc0 = nn.Linear(cfg.m_dim, cfg.num_relation, bias=False)
        self.proj_fc1 = nn.Linear(cfg.m_dim, cfg.num_relation, bias=False)
        self.proj_bias = nn.Parameter(torch.zeros(cfg.num_relation))

        # relatioin keys and values
        self.num_head = cfg.num_head
        self.tie_head_rel = cfg.tie_head_rel 
        r_dim = 2 * (cfg.m_dim // cfg.num_head if cfg.tie_head_rel else cfg.m_dim)
        relation_bank = torch.Tensor(cfg.num_relation, r_dim)
        self.relation_bank = nn.Parameter(relation_bank)
        nn.init.xavier_uniform_(self.relation_bank)

        self.use_only_pos = cfg.use_only_pos

    def forward(
        self, 
        x: Tensor, *args,
        bbox: Tensor=None,
        img_shape: Tuple=(320, 480),
        self_attn_mask: Tensor=None,
        self_key_padding_mask: Tensor=None,
        attn_weight_type: str=None,
        **kwargs
    ):
        if self.use_only_pos:
            x = torch.tensor(0., device=x.device)
        x = self._encode_positions(x, bbox, img_shape)

        x, _ = self.encoder(
            x, self_key_padding_mask=self_key_padding_mask, **kwargs,
        ) 
        
        # relation distribution prediction
        r_dist = ( 
            self.proj_fc0(x).unsqueeze(2) * 
            self.proj_fc1(x).unsqueeze(1) + 
            self.proj_bias[None, None, None]
        ) # (B, L, 1, R) * (B, 1, L, R) 

        r_dist = r_dist.softmax(dim=-1)

        B, L = x.shape[:2]
        relation_k, relation_v = (
            torch.matmul(r_dist, self.relation_bank)
                 .reshape(B, L, L, self.num_head, -1)
                 .permute(0, 3, 1, 2, 4)
                 .chunk(2, dim=-1)
        )
        return {"relation_k": relation_k, "relation_v": relation_v} 
