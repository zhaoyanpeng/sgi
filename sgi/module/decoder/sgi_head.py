import numpy as np
import os, sys, time
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerDecoder, TransformerDecoderLayer

from .. import PositionalEncoder, sign
from .. import MiniTF, MiniTFBlock, SignTFBlock, MiniTFAttention
from .. import RouteTFBlock, SpecialTFBlock, SpecialTFAttention

from .decoder_head import MiniTFDecHead


class RouteMiniTFDecHead(MiniTFDecHead):
    def __init__(self, cfg, token_vocab):
        super().__init__(cfg, token_vocab, skip_init=True)
        self.encoder = None
        if cfg.num_layer > 0:
            layer_fn = lambda ilayer: eval(cfg.block)(
                cfg.m_dim, cfg.num_head, cfg.f_dim, cfg.attn_cls_intra,
                attn_cls_inter=cfg.attn_cls_inter,
                ilayer=ilayer,
                dropout=cfg.t_dropout,
                qk_scale=cfg.qk_scale,
                activation=cfg.activation,
                attn_dropout=cfg.attn_dropout,
                proj_dropout=cfg.proj_dropout,
                self_ctx_dropout=cfg.route_self_dropout,
                memo_ctx_dropout=cfg.route_memo_dropout,
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


class SGIMiniTFMLMDecHead(RouteMiniTFDecHead):
    def __init__(self, cfg, token_vocab):
        super().__init__(cfg, token_vocab)
        # a special layer on top of the decoder
        self.post_encoder = SpecialTFBlock(
            cfg.m_dim, cfg.num_head, cfg.f_dim, cfg.attn_cls_intra, 
            attn_cls_inter="SpecialTFAttention", #cfg.attn_cls_inter, 
            ilayer=0, #ilayer,
            dropout=cfg.t_dropout, 
            qk_scale=cfg.qk_scale,
            activation=cfg.activation,
            attn_dropout=cfg.attn_dropout,
            proj_dropout=cfg.proj_dropout,
            self_ctx_dropout=cfg.infer_self_dropout,
            memo_ctx_dropout=cfg.infer_memo_dropout,
            num_head_intra=cfg.num_head_intra,
            num_head_inter=2, #cfg.num_head_inter,
            q_activation=cfg.q_activation,
            k_activation=cfg.k_activation,
            num_pos=cfg.max_dec_len,
            sign_q_intra=cfg.sign_q_intra,
            sign_k_intra=cfg.sign_k_intra,
            sign_q_inter=cfg.sign_q_inter,
            sign_k_inter=cfg.sign_k_inter,
            inter_layers=[], #list(cfg.inter_layers),
            routing=False,
        )
        self.split_vl = getattr(cfg, "split_vl", False)

    def forward(
        self,
        x: Tensor,
        memory: Tensor=None,
        memo_attn_mask: Tensor=None,
        memo_key_padding_mask: Tensor=None,
        mlm_inputs: Tensor=None,
        mlm_labels: Tensor=None,
        inter_attn_mask: Tensor=None,
        epoch_beta: float=0.,
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
            pass #return self.predictor(x), mlm_labels, {}

        attn_weights = list()
        if self.encoder is not None:
            x, attn_weights = self.encoder(
                x,
                memory=memory,
                self_attn_mask=None,
                memo_attn_mask=memo_attn_mask,
                self_key_padding_mask=self_key_padding_mask,
                memo_key_padding_mask=memo_key_padding_mask,
                require_attn_weight=True
            )

        x, attn_weights_ = self.post_encoder(
            x,
            memory=memory,
            self_attn_mask=None,
            memo_attn_mask=memo_attn_mask,
            self_key_padding_mask=self_key_padding_mask,
            memo_key_padding_mask=memo_key_padding_mask,
            require_attn_weight=True,
            split_vl=self.split_vl,
            epoch_beta=epoch_beta,
        )

        intra, (inter, logit_prior) = attn_weights_ # final layer
        attn_weights.append((intra, inter))

        """
        x = self.predictor(x)
        
        x_logit = x.log_softmax(dim=-1)
        if self.training:
            x = x_logit = (x_logit + logit_prior.unsqueeze(-1)).logsumexp(dim=2)
            # x.exp().sum(-1) should be all 1s
        else: # should never do inference here
            k = x_logit.shape[-1]
            indice = logit_prior.argmax(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, k)
            x = x_logit = x_logit.gather(2, indice).squeeze(2)
        """
        x_old = x
        x, _ = self.predict(x, logit_prior)
        
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
        logit_priors = list()
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

            attn_weights = list()
            if self.encoder is not None:
                x, attn_weights = self.encoder(
                    x,
                    memory=memory,
                    self_attn_mask=None,
                    memo_attn_mask=memo_attn_mask,
                    self_key_padding_mask=self_key_padding_mask,
                    memo_key_padding_mask=memo_key_padding_mask,
                    require_attn_weight=True
                )

            x, attn_weights_ = self.post_encoder(
                x,
                memory=memory,
                self_attn_mask=None,
                memo_attn_mask=memo_attn_mask,
                self_key_padding_mask=self_key_padding_mask,
                memo_key_padding_mask=memo_key_padding_mask,
                require_attn_weight=True,
                split_vl=self.split_vl,
            )

            intra, (inter, logit_prior) = attn_weights_ # final layer
            attn_weights.append((intra, inter))

            attn_weights = [
                [
                    attn[..., i, :] if attn is not None
                    else torch.tensor([], device=x.device) for attn in intra_inter
                ] for intra_inter in attn_weights
            ]
            attn_outs.append(attn_weights)
            dec_outs.append(x[:, i])
            logit_priors.append(logit_prior[:, i])

        nstep, nlayer, nattn = len(attn_outs), len(attn_outs[0]), len(attn_outs[0][0])
        attn_weights = [
            [
                torch.stack([
                    attn_outs[k][l][j] for k in range(nstep)
                ], -2) for j in range(nattn)
            ] for l in range(nlayer)
        ]
        x = torch.stack(dec_outs, 1)
        logit_prior = torch.stack(logit_priors, 1)

        """
        x = self.predictor(x)
        
        x_logit = x_logit_old = x.log_softmax(dim=-1)
        if self.training:
            x = x_logit = (x_logit + logit_prior.unsqueeze(-1)).logsumexp(dim=2)
            # x.exp().sum(-1) should be all 1s
        else:
            k = x_logit.shape[-1]
            indice = logit_prior.argmax(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, k)
            x = x_logit = x_logit.gather(2, indice).squeeze(2)
        """
        x, x_extra = self.predict(x, logit_prior)
        x_extra.update({"attns": attn_weights})

        mlm_labels = x_clone.clone()
        mlm_labels[special_token_masks] = -100
        return x, mlm_labels, x_extra

    def predict(self, x, logit_prior):
        x = self.predictor(x)

        x_extra = {}

        if x.shape[:3] == logit_prior.shape[:3]:
            x_logit = x.log_softmax(dim=-1)
        else:
            assert x.shape[2] - logit_prior.shape[2] == 1, f"illogical shapes {x.shape} {logit_prior.shape}"
            l_logit = x[:, :, :1]
            v_logit = x[:, :, 1:]

            alpha = 1. if self.training else .0

            #alpha, v_logit = 1., v_logit * 1.0

            x_logit = l_logit * alpha + v_logit
            x_logit = x_logit.log_softmax(dim=-1)

            x_extra["l_logit"] = l_logit
            x_extra["v_logit"] = v_logit

        x_extra["pair_logit"] = x_logit

        if self.training:
            x = x_logit = (x_logit + logit_prior.unsqueeze(-1)).logsumexp(dim=2)
            # x.exp().sum(-1) should be all 1s
        else:
            k = x_logit.shape[-1]
            indice = logit_prior.argmax(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, k)
            x = x_logit = x_logit.gather(2, indice).squeeze(2)

        return x, x_extra


class SGIMiniTFMLMLMDecHead(SGIMiniTFMLMDecHead):
    def __init__(self, cfg, token_vocab):
        super().__init__(cfg, token_vocab)
        self.lm_encoder = None
        if cfg.num_layer > 0:
            layer_fn = lambda ilayer: eval(cfg.block)(
                cfg.m_dim, cfg.num_head, cfg.f_dim, cfg.attn_cls_intra,
                attn_cls_inter=cfg.attn_cls_inter,
                ilayer=ilayer,
                dropout=cfg.t_dropout,
                qk_scale=cfg.qk_scale,
                activation=cfg.activation,
                attn_dropout=cfg.attn_dropout,
                proj_dropout=cfg.proj_dropout,
                self_ctx_dropout=cfg.route_self_dropout,
                memo_ctx_dropout=cfg.route_memo_dropout,
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
            self.lm_encoder = MiniTF(layer_fn, cfg.num_layer)

    def forward(
        self,
        x: Tensor,
        memory: Tensor=None,
        memo_attn_mask: Tensor=None,
        memo_key_padding_mask: Tensor=None,
        mlm_inputs: Tensor=None,
        mlm_labels: Tensor=None,
        inter_attn_mask: Tensor=None,
        epoch_beta: float=0.,
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
        
        original_x = x

        ##### MLM LOSS 

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
            pass #return self.predictor(x), mlm_labels, {}

        attn_weights = list()
        if self.encoder is not None:
            x, attn_weights = self.encoder(
                x,
                memory=memory,
                self_attn_mask=None,
                memo_attn_mask=memo_attn_mask,
                self_key_padding_mask=self_key_padding_mask,
                memo_key_padding_mask=memo_key_padding_mask,
                require_attn_weight=True
            )

        ### special cross attn
        x, attn_weights_ = self.post_encoder(
            x,
            memory=memory,
            self_attn_mask=None,
            memo_attn_mask=memo_attn_mask,
            self_key_padding_mask=self_key_padding_mask,
            memo_key_padding_mask=memo_key_padding_mask,
            require_attn_weight=True,
            split_vl=self.split_vl,
            epoch_beta=epoch_beta,
        )

        intra, (inter, logit_prior) = attn_weights_ # final layer
        attn_weights.append((intra, inter))

        x_old = x
        x_mlm, _ = self.predict(x, logit_prior)

        ##### LM LOSS 
        x = original_x 
        o_seqs = x[:, 1 :]
        i_seqs = x[:, :-1]
        length = i_seqs.size(1)

        self_key_padding_mask = (i_seqs == self.token_vocab.PAD_IDX)
        self_attn_mask = (torch.triu(
            torch.ones(length, length, dtype=torch.uint8, device=x.device), 
        diagonal=1) == 1)
        
        x = self._encode_positions(i_seqs)

        if self.lm_encoder is not None:
            x, attn_weights_lm = self.lm_encoder(
                x, 
                memory=memory, 
                self_attn_mask=self_attn_mask.squeeze(0),
                self_key_padding_mask=self_key_padding_mask,
                memo_key_padding_mask=memo_key_padding_mask,
                require_attn_weight=True
            ) 
        
        ### special cross attn
        x, attn_weights_lm_ = self.post_encoder(
            x,
            memory=memory,
            self_attn_mask=self_attn_mask.squeeze(0),
            memo_attn_mask=memo_attn_mask,
            self_key_padding_mask=self_key_padding_mask,
            memo_key_padding_mask=memo_key_padding_mask,
            require_attn_weight=True,
            split_vl=self.split_vl,
            epoch_beta=epoch_beta,
        )

        intra, (inter, logit_prior) = attn_weights_lm_ # final layer
        #attn_weights.append((intra, inter))

        #x_old = x
        x_lm, _ = self.predict(x, logit_prior)
        lm_labels = o_seqs.clone()
        lm_labels[o_seqs == self.token_vocab.PAD_IDX] = -100
        
        return x_mlm, mlm_labels, {"attns": attn_weights, "x_lm": x_lm, "lm_labels": lm_labels}
