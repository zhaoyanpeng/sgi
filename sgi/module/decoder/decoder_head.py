import numpy as np
import os, sys, time
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerDecoder, TransformerDecoderLayer

from fvcore.common.registry import Registry

from .. import PositionalEncoder, sign
from .. import MiniTF, MiniTFBlock, SignTFBlock, MiniTFAttention

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
        input_channels = wdim
        num_p, p_dim, cat_p, p_type = cfg.num_p, cfg.p_dim, cfg.cat_p, cfg.p_type
        if p_type == "sinuous":
            self.position_embed = PositionalEncoder(p_dim, dropout=cfg.p_dropout)
            if cat_p:
                input_channels += p_dim * 1
        elif p_type == "learned":
            self.position_embed = nn.Linear(num_p, p_dim, bias=False)
            if cat_p:
                input_channels += p_dim * 1
        else:
            self.register_parameter("position_embed", None)

        self.input_channels = input_channels
        if self.input_channels == cfg.m_dim:
            self.register_parameter("fc0", None)
        else:
            self.fc0 = nn.Linear(self.input_channels, cfg.m_dim, bias=False)
        self.ln0 = nn.LayerNorm(cfg.m_dim) if cfg.ln_input else nn.Identity()

        self._emb_size = cfg.m_dim
        self.cat_p = cat_p

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
            indices = torch.arange(x.shape[1], device=x.device)
            positions = self.position_embed.encode(indices).unsqueeze(0).expand(x.shape[0], -1, -1)
        elif isinstance(self.position_embed, nn.Linear):
            positions = self.position_embed.weight[:, :x.shape[1]]
            positions = positions.transpose(0, 1).unsqueeze(0).expand(x.shape[0], -1, -1)
        if self.cat_p:
            x = torch.cat([x, positions], -1)
        else:
            assert list(x.shape) == [] or x.shape == positions.shape
            x = x + positions
        x = self.fc0(x) if self.fc0 is not None else x
        x = self.ln0(x)
        return x

@DECODER_HEADS_REGISTRY.register()
class MiniTFDecHead(MetaDecHead):
    def __init__(self, cfg, token_vocab):
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
                num_pos=cfg.max_dec_len,
                sign_q_intra=cfg.sign_q_intra,
                sign_k_intra=cfg.sign_k_intra,
                sign_q_inter=cfg.sign_q_inter,
                sign_k_inter=cfg.sign_k_inter,
                inter_layers=list(cfg.inter_layers),
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
        self.encoder = None
        if cfg.num_layer > 0:
            layer_fn = TransformerDecoderLayer(
                cfg.m_dim, cfg.num_head, cfg.f_dim, cfg.t_dropout, activation=cfg.activation,
            )
            self.encoder = TransformerDecoder(layer_fn, cfg.num_layer)

        self.num_head = cfg.num_head

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
            tgt_mask=self_attn_mask,
            tgt_key_padding_mask=self_key_padding_mask,
            memory_key_padding_mask=memo_key_padding_mask,
        ) 

        x = x.transpose(0, 1)

        x = self.predictor(x) 
        return x, o_seqs.contiguous(), {}

@DECODER_HEADS_REGISTRY.register()
class TorchTFMLMDecHead(TorchTFDecHead):
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

        x = self._encode_positions(i_seqs)

        if self.encoder is None:
            return self.predictor(x), mlm_labels, {}


        self_key_padding_mask = (i_seqs == self.token_vocab.PAD_IDX)

        if inter_attn_mask is not None: # hard attention provided externally
            if inter_attn_mask.dim() == 2: # (B, S)
                memo_key_padding_mask = inter_attn_mask | memo_key_padding_mask
            elif inter_attn_mask.dim() == 3: # (B, T, S)
                assert memo_attn_mask is None
                B, T, S = inter_attn_mask.shape[:3]
                memo_attn_mask = inter_attn_mask.unsqueeze(1).expand(-1, self.num_head, -1, -1).reshape(-1, T, S)


        memory = memory.transpose(0, 1)

        x = x.transpose(0, 1)
        
        x = self.encoder(
            x,
            memory=memory,
            tgt_mask=None,
            memory_mask=memo_attn_mask,
            tgt_key_padding_mask=self_key_padding_mask,
            memory_key_padding_mask=memo_key_padding_mask,
        )

        x = x.transpose(0, 1)

        x = self.predictor(x)
        return x, mlm_labels, {"attns": None}

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
                B, T, S = inter_attn_mask.shape[:3]
                memo_attn_mask = inter_attn_mask.unsqueeze(1).expand(-1, self.num_head, -1, -1).reshape(-1, T, S)

        memory = memory.transpose(0, 1)

        for i in range(L):
            mlm_inputs = x_clone.clone()
            #mlm_labels = x_clone.clone()
            #mlm_labels[:, :i] = -100
            #mlm_labels[:, i + 1:] = -100

            #mlm_labels[:, i] = mlm_inputs[:, i]
            #mlm_labels[special_token_masks] = -100

            mlm_inputs[:, i] = self.token_vocab("<mask>")
            #mlm_inputs[self_key_padding_mask] = self.token_vocab.PAD_IDX

            i_seqs = mlm_inputs

            x = self._encode_positions(i_seqs)

            x = x.transpose(0, 1)

            x = self.encoder(
                x,
                memory=memory,
                tgt_mask=None,
                memory_mask=memo_attn_mask,
                tgt_key_padding_mask=self_key_padding_mask,
                memory_key_padding_mask=memo_key_padding_mask,
            )

            x = x.transpose(0, 1)

            dec_outs.append(x[:, i])

        x = torch.stack(dec_outs, -2)

        x = self.predictor(x)
        mlm_labels = x_clone.clone()
        mlm_labels[special_token_masks] = -100
        return x, mlm_labels, {"attns": None}

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
            #mlm_inputs[self_key_padding_mask] = self.token_vocab.PAD_IDX

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
                [
                    attn[..., i, :] if attn is not None
                    else torch.tensor([], device=x.device) for attn in intra_inter
                ] for intra_inter in attn_weights
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


@DECODER_HEADS_REGISTRY.register()
class ToyTFMLMDecHead(MetaDecHead):
    def __init__(self, cfg, token_vocab):
        super().__init__(cfg, token_vocab)
        self.case = 4
        if self.case in {2, 3}:
            self.q_proj = nn.Linear(cfg.m_dim, cfg.m_dim, bias=True)
            self.k_proj = nn.Linear(cfg.m_dim, cfg.m_dim, bias=True)
        elif self.case in {5}:
            self.q_proj = nn.Linear(cfg.m_dim, cfg.m_dim, bias=True)
            self.k_proj = nn.Linear(cfg.m_dim, cfg.m_dim, bias=True)
            self.linear = nn.Linear(cfg.m_dim * 2, cfg.m_dim, bias=True)
        elif self.case in {4}:
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
                    sign_q_intra=cfg.sign_q_intra,
                    sign_k_intra=cfg.sign_k_intra,
                    sign_q_inter=cfg.sign_q_inter,
                    sign_k_inter=cfg.sign_k_inter,
                    inter_layers=list(cfg.inter_layers),
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
        mlm_inputs: Tensor=None,
        mlm_labels: Tensor=None,
        inter_attn_mask: Tensor=None,
        infer: bool=False,
        **kwargs,
    ):
        if memory is None: # may or may not have inter attention
            memory = memo_attn_mask = memo_key_padding_mask = None
        if infer and not self.training and self.case == 4:
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

        B, S, H = memory.shape

        i_seqs = mlm_inputs
        self_key_padding_mask = (i_seqs == self.token_vocab.PAD_IDX)

        if inter_attn_mask is not None: # hard attention provided externally
            if inter_attn_mask.dim() == 2: # (B, S)
                memo_key_padding_mask = inter_attn_mask | memo_key_padding_mask
            elif inter_attn_mask.dim() == 3: # (B, T, S)
                assert memo_attn_mask is None
                memo_attn_mask = inter_attn_mask

        ###############################
        # BEGIN (signed) weighted sum of memo
        ###############################
        
        if self.case not in {4}:
            #print(sign_weight, sign_weight.shape)
            inter_map = ~inter_attn_mask[:, [1, 3]]
            #print(inter_map)
            shape = inter_map.shape[:2] + (-1,)
            memo = memory.unsqueeze(1).masked_select(
                inter_map.unsqueeze(-1)
            ).reshape(shape)
            #m = memo[:, :, 256:260]
            #print(m)

            #memo[..., :256] = 0.

            #indice = torch.nonzero(inter_map.int())
            #indice = indice[:, -1].reshape(B, -1)
            #print(indice)

        ###############################
        # END (signed) weighted sum of memo
        ###############################

        x = self._encode_positions(i_seqs)

        case = self.case
        reset_wemb = False

        if case == 1:
            if reset_wemb:
                positions = self.position_embed.weight[:, [1, 2, 3]]
                positions = positions.transpose(0, 1).unsqueeze(0).expand(B, -1, -1)
                x[:, [1, 3]] = positions[:, [0, 2]]
                #x[:, [1, 2, 3]] = positions

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

            sign_weight = attn_weights[0][0].squeeze()
            attn_weights = sign_weight = sign_weight[:, 2, [1, 3]]
            x = sign_weight.unsqueeze(1) @ memo
            mlm_labels = mlm_labels[:, 2:3]
            #print(mlm_labels)
        elif case == 2: # almost the same as case 3
            positions = self.position_embed.weight[:, [1, 2, 3]]
            positions = positions.transpose(0, 1).unsqueeze(0).expand(B, -1, -1)
            x[:, [1, 3]] = positions[:, [0, 2]]
            #x[:, [1, 2, 3]] = positions
            positions = x[:, [1, 2, 3]]

            sq = self.q_proj(positions)[:, 1:2]
            sk = self.k_proj(positions)[:, [0, 2]]
            #sq = sq.relu()
            #sk = sk.relu()
            sign_weight = sq @ sk.transpose(-1, -2)
            act = nn.Tanh() #nn.Softsign() #
            sign_weight = act(sign_weight)
            #sign_weight = sign(sign_weight)
            #sign_weight = torch.tensor([[-1., 1.]], device=x.device).expand(B, -1)
            attn_weights = sign_weight = sign_weight.squeeze()
            #print(sign_weight)
            x = sign_weight.unsqueeze(1) @ memo
            mlm_labels = mlm_labels[:, 2:3]
            #print(mlm_labels)
        elif case == 3:
            positions = self.position_embed.weight[:, [1, 2, 3]]
            positions = positions.transpose(0, 1)

            sq = self.q_proj(positions)[1:2]
            sk = self.k_proj(positions)[[0, 2]]
            #sq = sq.relu()
            #sk = sk.relu()
            sign_weight = sq @ sk.transpose(-1, -2)
            act = nn.Tanh() #nn.Softsign() #
            sign_weight = act(sign_weight)
            #sign_weight = sign(sign_weight)
            #sign_weight = torch.tensor([[-1., 1.]], device=x.device).expand(B, -1)
            attn_weights = sign_weight = sign_weight.expand(B, -1)
            #print(sign_weight)
            x = sign_weight.unsqueeze(1) @ memo
            mlm_labels = mlm_labels[:, 2:3]
            #print(mlm_labels)
        elif case == 4: # can it learn queries?
            if self.encoder is None:
                return self.predictor(x), mlm_labels, {}

            #self_key_padding_mask = torch.full(i_seqs.shape, 1, device=i_seqs.device).bool()
            #self_key_padding_mask[:, [1, 2, 3]] = False

            x, attn_weights = self.encoder(
                x,
                memory=memory,
                self_attn_mask=None,
                memo_attn_mask=memo_attn_mask,
                self_key_padding_mask=self_key_padding_mask,
                memo_key_padding_mask=memo_key_padding_mask,
                require_attn_weight=True
            )

            #sign_weight = attn_weights[-1][1].squeeze() # inter attn
            #attn_weights = sign_weight = sign_weight[..., 2, [1, 3]]
            #x = x[:, 2:3]
            #mlm_labels = mlm_labels[:, 2:3]
            #print(mlm_labels)
        elif case == 5: # almost the same as case 3
            positions = x[:, [1, 3]].reshape(B, 1, -1)
            positions = self.linear(positions)

            sq = self.q_proj(positions)
            sk = self.k_proj(memo)
            q_act = nn.GELU() #nn.ReLU() #nn.CELU() #nn.Identity() # nn.Softsign() #
            k_act = nn.GELU() #nn.ReLU() #nn.CELU() #nn.Identity() # nn.Softsign() #
            sq = q_act(sq)
            sk = k_act(sk)
            sign_weight = sq @ sk.transpose(-1, -2)
            act = nn.Tanh() #nn.Softsign() #
            sign_weight = act(sign_weight)
            #sign_weight = sign(sign_weight)
            #sign_weight = torch.tensor([[-1., 1.]], device=x.device).expand(B, -1)
            attn_weights = sign_weight = sign_weight.squeeze()
            #print(sign_weight)
            x = sign_weight.unsqueeze(1) @ memo
            mlm_labels = mlm_labels[:, 2:3]
            #print(mlm_labels)

        #print(x[:, :, 256:260])
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

        self_key_padding_mask = torch.full(x.shape, 1, device=x.device).bool()
        self_key_padding_mask[:, [1, 2, 3]] = False

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
