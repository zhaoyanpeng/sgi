import math
import torch
from torch import nn

from gensim.test.utils import datapath
from gensim.models import KeyedVectors

__all__ = [
    "SelfattentionMask",
    "PositionalEncoder", 
    "PartiallyFixedEmbedding",
]

class SelfattentionMask(nn.Module):
    """
    A module which provides self-attention masks, typically used in a decoder.
    """
    def __init__(self, device=None, max_len=256):
        super(SelfattentionMask, self).__init__()
        self.make_mask(max_len, device)

    def make_mask(self, max_len, device):
        assert max_len >= 0
        self.device = device
        self.max_len = max_len

        weight = torch.ones(
            (max_len, max_len), dtype=torch.uint8 #.bool #, device=device
        ).triu_(1)
        self.register_buffer("weight", weight)
    
    def forward(self, length, device=None):
        assert length >= 0
        if length > self.max_len:
            self.make_mask(length, device)
        device = self.device if device is None else device
        weight = self.weight[: length, : length].to(device)
        return weight.unsqueeze(0) 

class PositionalEncoder(nn.Module):
    """
    A position encoder described as in the Transformer paper.
    It provides two typical implementations of the model:
    (1) one is the same as the model in the paper;
    (2) the other is the implementation in Tensor2tensor.
    """
    def __init__(self, D, dropout=0.1, device=None, max_len=1024, use_tf=True):
        super(PositionalEncoder, self).__init__()
        self.use_tf = use_tf
        self.make_pe(D, max_len, device)
        self.dropout = nn.Dropout(p=dropout)

    def make_pe(self, D, max_len, device):
        if self.use_tf:
            self.make_pe_tf(D, max_len, device) 
        else:
            self.make_pe_tt(D, max_len, device) 

    def make_pe_tf(self, D, max_len, device, min_timescale=1.0, max_timescale=1e4):
        """
        The standard Transformer implementation (interleave).
        """
        self.D = D
        self.device = device
        self.max_len = max_len

        assert D % 2 == 0 #FIXME assuming D must be a multiple of 2

        div_term = torch.exp(
            torch.arange(0, D, 2).float() * (-math.log(max_timescale) / D)
        )
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        weight = torch.zeros(max_len, D)
        weight[:, 0 : : 2] = torch.sin(position * div_term) 
        weight[:, 1 : : 2] = torch.cos(position * div_term)
        weight = weight.unsqueeze(0) #.to(device)
        self.register_buffer("weight", weight)

    def make_pe_tt(self, D, max_len, device, min_timescale=1.0, max_timescale=1e4):
        """
        Tensor2tensor implementation (concatenate). 

        There should be nothing different from the standard implementation.
        See `https://github.com/tensorflow/tensor2tensor/blob/ba8c10d770eda18594520dc91f84e54fe15a3fa6/tensor2tensor/layers/common_attention.py#L653`.
        """
        self.D = D
        self.device = device
        self.max_len = max_len

        dim = D // 2
        div_term = -math.log(max_timescale) / min_timescale / float(D - 1)
        dimension = min_timescale * torch.exp(
            torch.arange(0, dim, dtype=torch.float) * div_term
        ).unsqueeze(0)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1) 
        weight = position * dimension 
        weight = torch.cat([torch.sin(weight), torch.cos(weight)], dim=1)
        if D % 2 == 1:
            weight = torch.cat([weight, torch.zeros(max_len, 1)], dim=1) 
        weight = weight.unsqueeze(0) #.to(device)
        self.register_buffer("weight", weight)

    def forward(self, x, offset=0, batch_first=True):
        """
        Args:
            x (Tensor): batch-first tensor as (B, L, D)
        """
        seq_len = x.size(1) if batch_first else x.size(0)
        if seq_len + offset > self.max_len: 
            self.make_pe(self.D, seq_len + offset, x.device)
        indice = offset + torch.arange(seq_len)
        weight = self.weight[:, indice, :].to(x.device)
        if not batch_first:
            assert weight.size(1) == seq_len
            weight = weight.transpose(0, 1)
        return self.dropout(x + weight)

    def encode(self, x):
        """
        Args:
            x (Tensor): batch-first tensor as (B, L, D)
        """
        shape = x.shape + (-1,)
        weight = self.weight[:, x.reshape(-1), :].to(x.device)
        return weight.squeeze(0).view(shape)

class BboxRelationalEncoder(nn.Module):
    """
    Given a tensor with bbox coordinates for detected objects on each batch image,
    this function computes a matrix for each image
    with entry (i,j) given by a vector representation of the
    displacement between the coordinates of bbox_i, and bbox_j
    input: np.array of shape=(batch_size, max_nr_bounding_boxes, 4)
    output: np.array of shape=(batch_size, max_nr_bounding_boxes, max_nr_bounding_boxes, 64)
    """
    #returns a relational embedding for each pair of bboxes, with dimension = dim_g
    #follow implementation of https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1014-L1055
    def __init__(self, dim_g=64, wave_len=1000, trignometric_embedding=True):
        super(BboxRelationalEncoder, self).__init__()
        self.trignometric_embedding = trignometric_embedding
        self.wave_len = wave_len
        self.dim_g = dim_g

    def forward(self, f_g):
        batch_size = f_g.size(0)

        x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=-1)

        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5
        w = (x_max - x_min) + 1.
        h = (y_max - y_min) + 1.

        #cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
        delta_x = cx - cx.view(batch_size, 1, -1)
        delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
        delta_x = torch.log(delta_x)

        delta_y = cy - cy.view(batch_size, 1, -1)
        delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
        delta_y = torch.log(delta_y)

        delta_w = torch.log(w / w.view(batch_size, 1, -1))
        delta_h = torch.log(h / h.view(batch_size, 1, -1))

        matrix_size = delta_h.size()
        delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
        delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
        delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
        delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

        position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)

        if self.trignometric_embedding == True:
            feat_range = torch.arange(self.dim_g / 8).cuda()
            dim_mat = feat_range / (self.dim_g / 8)
            dim_mat = 1. / (torch.pow(self.wave_len, dim_mat))

            dim_mat = dim_mat.view(1, 1, 1, -1)
            position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
            position_mat = 100. * position_mat

            mul_mat = position_mat * dim_mat
            mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
            sin_mat = torch.sin(mul_mat)
            cos_mat = torch.cos(mul_mat)
            embedding = torch.cat((sin_mat, cos_mat), -1)
        else:
            embedding = position_mat
        return(embedding)

class PartiallyFixedEmbedding(torch.nn.Module):
    def __init__(self, vocab, w2vec_file, word_dim=-1, out_dim=-1):
        super(PartiallyFixedEmbedding, self).__init__()
        nword = len(vocab)
        model = KeyedVectors.load_word2vec_format(datapath(w2vec_file), binary=False)
        masks = [1 if vocab.idx2word[k] in model.vocab else 0 for k in range(nword)]
        idx2fixed = [k for k in range(nword) if masks[k]]
        idx2tuned = [k for k in range(nword) if not masks[k]]
        arranged_idx = idx2fixed + idx2tuned
        idx_mapping = {idx: real_idx for real_idx, idx in enumerate(arranged_idx)}
        self.register_buffer("realid", torch.tensor(
            [idx_mapping[k] for k in range(nword)], dtype=torch.int64
        ))
        self.idx_mapping = idx_mapping
        self.n_fixed = sum(masks)
        n_tuned = nword - self.n_fixed

        weight = torch.empty(nword, model.vector_size)
        for k, idx in vocab.word2idx.items():
            real_idx = idx_mapping[idx]
            if k in model.vocab:
                weight[real_idx] = torch.tensor(model[k])

        self.tuned_weight = torch.nn.Parameter(torch.empty(n_tuned, model.vector_size)) 
        torch.nn.init.kaiming_uniform_(self.tuned_weight)
        weight[self.n_fixed:] = self.tuned_weight
        self.register_buffer("weight", weight)
         
        add_dim = word_dim - model.vector_size if word_dim > model.vector_size else 0 
        self.tuned_vector = torch.nn.Parameter(torch.empty(nword, add_dim))
        if add_dim > 0: 
            torch.nn.init.kaiming_uniform_(self.tuned_vector)
        in_dim = model.vector_size if add_dim == 0 else word_dim 

        self.linear = (
            torch.nn.Linear(in_dim, out_dim, bias=False)
            if out_dim > 0 else torch.nn.Identity() #None #lambda x: x
        )
        self._output_dim = out_dim if out_dim > 0 else in_dim
        del model

    @property
    def output_dim(self):
        return self._output_dim

    def __setstate__(self, state):
        super(PartiallyFixedEmbedding, self).__setstate__(state)
        pass

    def extra_repr(self):
        mod_keys = self._modules.keys()
        all_keys = self._parameters.keys()
        extra_keys = all_keys - mod_keys
        extra_keys = [k for k in all_keys if k in extra_keys]
        extra_lines = []
        for key in extra_keys:
            attr = getattr(self, key)
            if not isinstance(attr, torch.nn.Parameter):
                continue
            extra_lines.append("({}): Tensor{}".format(key, tuple(attr.size())))
        return "\n".join(extra_lines)

    def reindex(self, X):
        return X.clone().cpu().apply_(self.idx_mapping.get)

    def bmm(self, X):
        self.realid.detach_()
        self.weight.detach_()
        self.weight[self.n_fixed:] = self.tuned_weight
        word_emb = torch.cat([self.weight, self.tuned_vector], -1)
        word_emb = word_emb[self.realid] 
        word_emb = self.linear(word_emb)
        x_shape = X.size()
        w_logit = torch.matmul(
            X.view(-1, x_shape[-1]), word_emb.transpose(0, 1)
        )
        w_logit = w_logit.view(x_shape[:-1] + (w_logit.size(-1),))
        return w_logit 

    def forward(self, X):
        if X.dtype != torch.int64:  
            return self.bmm(X) # w/o linear 
        self.weight.detach_()
        self.weight[self.n_fixed:] = self.tuned_weight
        weight = torch.cat([self.weight, self.tuned_vector], -1)
        #X = X.clone().cpu().apply_(self.idx_mapping.get) # only work on cpus
        X = X.clone().cpu().apply_(self.idx_mapping.get).cuda() 
        #print(X, weight.device, self.weight.device, self.tuned_vector.device)
        word_vect = torch.nn.functional.embedding(X, weight, None, None, 2.0, False, False)
        if self.linear is not None:
            word_vect = self.linear(word_vect)
        return word_vect

