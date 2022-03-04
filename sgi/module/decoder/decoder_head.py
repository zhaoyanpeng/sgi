import numpy as np
import os, sys, time
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerDecoder, TransformerDecoderLayer

from fvcore.common.registry import Registry

from .. import PositionalEncoder
from .. import MiniTF, MiniTFBlock, MiniTFAttention

DECODER_HEADS_REGISTRY = Registry("DECODER_HEADS")
DECODER_HEADS_REGISTRY.__doc__ = """
Registry for decoder heads.
"""

initializr = lambda x: None 

def build_decoder_head(cfg, vocab):
    return DECODER_HEADS_REGISTRY.get(cfg.name)(cfg, vocab)

class MetaDecHead(nn.Module):
    def __init__(self, cfg, token_vocab):
        super(MetaDecHead, self).__init__()
        # input embedding layer
        wdim = cfg.w_dim
        self.token_embed = None
        self.token_vocab = token_vocab
        if os.path.isfile(cfg.w2v_file):
            self.token_embed = PartiallyFixedEmbedding(self.token_vocab, cfg.w2v_file)
        elif not cfg.skip_embed:
            self.token_embed = nn.Embedding(
                len(self.token_vocab), wdim, padding_idx=self.token_vocab.PAD_IDX
            )
        # input positional embedding 
        num_p, p_dim, p_type = cfg.num_p, cfg.p_dim, cfg.p_type
        if p_type == "sinuous":
            self.position_embed = PositionalEncoder(p_dim, dropout=cfg.p_dropout)
        elif p_type == "learned":
            self.position_embed = nn.Linear(num_p, p_dim, bias=False)
        else:
            self.register_parameter("position_embed", None)

        if cfg.w_dim == cfg.m_dim:
            self.register_parameter("fc0", None)
        else:
            self.fc0 = nn.Linear(cfg.w_dim, cfg.m_dim, bias=False) 
        self.ln0 = nn.LayerNorm(cfg.m_dim) if cfg.ln_input else nn.Identity()

        self._emb_size = cfg.m_dim

    @property
    def emb_size(self):
        return self._emb_size

    def init_state_from_memo(self, memo, mask, **args):
        pass

    def detach_state(self):
        pass

    def init_state(self, encoder_final, **args):
        pass

    def _reset_parameters(self):
        for field in [self.fc0]:
            if field is None: continue
            if hasattr(field, "weight") and field.weight is not None:
                initializr(field.weight) 
            if hasattr(field, "bias") and field.bias is not None:
                nn.init.constant_(field.bias, 0.)

    def _encode_positions(self, x):
        if self.token_embed is not None:
            x = self.token_embed(x)
        if isinstance(self.position_embed, PositionalEncoder):
            x = self.position_embed(x)
        elif isinstance(self.position_embed, nn.Linear):
            positions = self.position_embed.weight[:, :x.shape[1]]
            positions = positions.transpose(0, 1).unsqueeze(0)
            x += positions
        x = self.fc0(x) if self.fc0 is not None else x
        x = self.ln0(x)
        return x

@DECODER_HEADS_REGISTRY.register()
class MiniTFDecHead(MetaDecHead):
    def __init__(self, cfg, token_vocab):
        super().__init__(cfg, token_vocab)
        self.encoder = None
        if cfg.num_layer > 0: 
            layer_fn = lambda : MiniTFBlock(
                cfg.m_dim, cfg.num_head, cfg.f_dim, cfg.attn_cls_intra, 
                attn_cls_inter=cfg.attn_cls_inter, 
                dropout=cfg.t_dropout, 
                qk_scale=cfg.qk_scale,
                activation=cfg.activation,
                attn_dropout=cfg.attn_dropout,
                proj_dropout=cfg.proj_dropout,
                num_head_intra=cfg.num_head_intra,
                num_head_inter=cfg.num_head_inter,
                sign_q=cfg.sign_q,
                sign_k=cfg.sign_k,
            )
            self.encoder = MiniTF(layer_fn, cfg.num_layer) 

        self.predictor = nn.Sequential(
            nn.Linear(cfg.m_dim, len(self.token_vocab))
        ) 

        self.max_dec_len = cfg.max_dec_len
        self._reset_parameters()

    def forward( 
        self, 
        x: Tensor,
        memory: Tensor=None,
        memo_attn_mask: Tensor=None,
        memo_key_padding_mask: Tensor=None,
        **kwargs,
    ):
        if memory is None: # may or may not have inter attention
            memory = memo_attn_mask = memo_key_padding_mask = None 
        if False and not self.training:
            return self.inference(
                x,
                memory=memory,
                memo_attn_mask=memo_attn_mask,
                memo_key_padding_mask=memo_key_padding_mask
            )

        o_seqs = x[:, 1 :]
        i_seqs = x[:, :-1]
        length = i_seqs.size(1)

        self_key_padding_mask = (i_seqs == self.token_vocab.PAD_IDX)
        self_attn_mask = (torch.triu(
            torch.ones(length, length, dtype=torch.uint8, device=x.device), 
        diagonal=1) == 1)
        
        x = self._encode_positions(i_seqs)

        if self.encoder is None:
            return self.predictor(x), o_seqs.contiguous(), {} 
        
        x, attn_weights = self.encoder(
            x, 
            memory=memory, 
            self_attn_mask=self_attn_mask.squeeze(0),
            self_key_padding_mask=self_key_padding_mask,
            memo_key_padding_mask=memo_key_padding_mask,
            require_attn_weight=True
        ) 

        x = self.predictor(x) 
        return x, o_seqs.contiguous(), {"attns": attn_weights}

    def inference(
        self, 
        x: Tensor,
        memory: Tensor = None,
        memo_attn_mask: Tensor = None,
        memo_key_padding_mask: Tensor = None,
        **kwargs,
    ):
        # to generate sequences
        o_seqs = x[:, 1 :]
        i_seqs = x[:, :-1]
        length = i_seqs.size(1)

        beg_len = 0 
        logits = list() 
        if beg_len > 0:
            all_ctx = i_seqs[:, :beg_len + 1]
            logit = torch.zeros((all_ctx.size(0), beg_len, len(self.token_vocab)), device=all_ctx.device)
            logit = logit.scatter(2, all_ctx[:, 1:].unsqueeze(-1), 10)
            logits.append(logit)
        else:
            all_ctx = i_seqs[:, :1]

        for i in range(beg_len, self.max_dec_len):
            x = self._encode_positions(all_ctx)

            x, _ = self.encoder(
                x, 
                memory=memory, 
                memo_key_padding_mask=memo_key_padding_mask,
            ) 

            logit = self.predictor(x[:, -1:])
            logits.append(logit)

            new_ctx = logit.argmax(dim=-1)
            all_ctx = torch.cat((all_ctx, new_ctx), 1)

        all_logits = torch.cat(logits, dim=1)
        return all_logits, o_seqs, {}

@DECODER_HEADS_REGISTRY.register()
class TorchTFDecHead(MetaDecHead):
    def __init__(self, cfg, token_vocab):
        super().__init__(cfg, token_vocab)
        layer_fn = TransformerDecoderLayer(
            cfg.m_dim, cfg.num_head, cfg.f_dim, cfg.t_dropout, activation=cfg.activation, 
        ) 
        self.encoder = TransformerDecoder(layer_fn, cfg.num_layer) 

        self.predictor = nn.Sequential(
            nn.Linear(cfg.m_dim, len(self.token_vocab))
        ) 

        self.max_dec_len = cfg.max_dec_len
        self._reset_parameters()

    def forward( 
        self, 
        x: Tensor,
        memory: Tensor=None,
        memo_attn_mask: Tensor=None,
        memo_key_padding_mask: Tensor=None,
        **kwargs,
    ):
        if memory is None: # may or may not have inter attention
            memory = memo_attn_mask = memo_key_padding_mask = None 
        if False and not self.training:
            return self.inference(
                x,
                memory=memory,
                memo_attn_mask=memo_attn_mask,
                memo_key_padding_mask=memo_key_padding_mask
            )

        o_seqs = x[:, 1 :]
        i_seqs = x[:, :-1]
        length = i_seqs.size(1)

        self_key_padding_mask = (i_seqs == self.token_vocab.PAD_IDX)
        self_attn_mask = (torch.triu(
            torch.ones(length, length, dtype=torch.uint8, device=x.device), 
        diagonal=1) == 1)
        
        x = self._encode_positions(i_seqs)

        memory = memory.transpose(0, 1)

        x = x.transpose(0, 1)
        
        x = self.encoder(
            x, 
            memory=memory, 
            tgt_mask=self_attn_mask.squeeze(0),
            tgt_key_padding_mask=self_key_padding_mask,
            memory_key_padding_mask=memo_key_padding_mask,
        ) 

        x = x.transpose(0, 1)

        x = self.predictor(x) 
        return x, o_seqs.contiguous(), {}

@DECODER_HEADS_REGISTRY.register()
class MiniTFMLMDecHead(MiniTFDecHead):
    def forward(
        self,
        x: Tensor,
        memory: Tensor=None,
        memo_attn_mask: Tensor=None,
        memo_key_padding_mask: Tensor=None,
        mlm_inputs: Tensor=None,
        mlm_labels: Tensor=None,
        inter_attn_mask: Tensor=None,
        infer: bool=False,
        **kwargs,
    ):
        if memory is None: # may or may not have inter attention
            memory = memo_attn_mask = memo_key_padding_mask = None
        if infer and not self.training:
            return self.inference(
                x,
                memory=memory,
                memo_attn_mask=memo_attn_mask,
                memo_key_padding_mask=memo_key_padding_mask,
                mlm_inputs=mlm_inputs,
                mlm_labels=mlm_labels,
                inter_attn_mask=inter_attn_mask,
                **kwargs,
            )

        i_seqs = mlm_inputs
        self_key_padding_mask = (i_seqs == self.token_vocab.PAD_IDX)

        if inter_attn_mask is not None: # hard attention provided externally
            if inter_attn_mask.dim() == 2: # (B, S)
                memo_key_padding_mask = inter_attn_mask | memo_key_padding_mask
            elif inter_attn_mask.dim() == 3: # (B, T, S)
                assert memo_attn_mask is None
                memo_attn_mask = inter_attn_mask

        x = self._encode_positions(i_seqs)

        if self.encoder is None:
            return self.predictor(x), mlm_labels, {} 

        x, attn_weights = self.encoder(
            x,
            memory=memory,
            self_attn_mask=None,
            memo_attn_mask=memo_attn_mask,
            self_key_padding_mask=self_key_padding_mask,
            memo_key_padding_mask=memo_key_padding_mask,
            require_attn_weight=True
        )

        x = self.predictor(x)
        return x, mlm_labels, {"attns": attn_weights}

    def inference(
        self,
        x: Tensor,
        memory: Tensor = None,
        memo_attn_mask: Tensor = None,
        memo_key_padding_mask: Tensor = None,
        mlm_inputs: Tensor=None,
        mlm_labels: Tensor=None,
        inter_attn_mask: Tensor=None,
        **kwargs,
    ):
        device = x.device
        B, L = x.shape[:2]

        #batch_indice = torch.arange(B, device=device)
        self_key_padding_mask = (x == self.token_vocab.PAD_IDX)
        special_token_masks = self.token_vocab.get_special_token_masks(x)

        dec_outs = list()
        attn_outs = list()
        x_clone = x.clone()

        if inter_attn_mask is not None: # hard attention provided externally
            if inter_attn_mask.dim() == 2: # (B, S)
                memo_key_padding_mask = inter_attn_mask | memo_key_padding_mask
            elif inter_attn_mask.dim() == 3: # (B, T, S)
                assert memo_attn_mask is None
                memo_attn_mask = inter_attn_mask

        for i in range(L):
            mlm_inputs = x_clone.clone()
            #mlm_labels = x_clone.clone()
            #mlm_labels[:, :i] = -100
            #mlm_labels[:, i + 1:] = -100

            #mlm_labels[:, i] = mlm_inputs[:, i]
            #mlm_labels[special_token_masks] = -100

            mlm_inputs[:, i] = self.token_vocab("<mask>")
            mlm_inputs[self_key_padding_mask] = self.token_vocab.PAD_IDX

            i_seqs = mlm_inputs

            x = self._encode_positions(i_seqs)

            x, attn_weights = self.encoder(
                x,
                memory=memory,
                self_attn_mask=None,
                memo_attn_mask=memo_attn_mask,
                self_key_padding_mask=self_key_padding_mask,
                memo_key_padding_mask=memo_key_padding_mask,
                require_attn_weight=True
            )

            attn_weights = [
                [attn[..., i, :] for attn in intra_inter] for intra_inter in attn_weights
            ]
            attn_outs.append(attn_weights)
            dec_outs.append(x[:, i])

        nstep, nlayer, nattn = len(attn_outs), len(attn_outs[0]), len(attn_outs[0][0])
        attn_weights = [
            [
                torch.stack([
                    attn_outs[k][l][j] for k in range(nstep)
                ], -2) for j in range(nattn)
            ] for l in range(nlayer)
        ]
        x = torch.stack(dec_outs, -2)

        x = self.predictor(x)
        mlm_labels = x_clone.clone()
        mlm_labels[special_token_masks] = -100
        return x, mlm_labels, {"attns": attn_weights}
