# Copyright (c) Mr. Robot, Inc. and its affiliates. All Rights Reserved
import os
import copy
import torch
import logging
import numpy as np
import pickle
import random
import warnings

from collections import defaultdict

from .catalog import DatasetCatalog

__all__ = ["Indexer", "register_indexer", "mask_tokens", "SortedSequentialSampler"]


class SortedSequentialSampler(torch.utils.data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        self.data_source._shuffle()
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)

class Indexer:
    """
    Build vocabulary from a pre-defined word-index map.

    Args:
        index_file: a file containing <word, index> per line.
            Indices must be a contiguous `int` sequence. The 
            first four words must be `PAD`,`UNK`,`BOS`,`EOS`.
    """
    def __init__(self, index_file=None, extra_keys=[], name=""):
        self.PAD, self.UNK, self.BOS, self.EOS = ["<pad>", "<unk>", "<s>", "</s>"]
        self.PAD_IDX, self.UNK_IDX, self.BOS_IDX, self.EOS_IDX = [0, 1, 2, 3]
        self.word2idx = {
            self.PAD: self.PAD_IDX, 
            self.UNK: self.UNK_IDX, 
            self.BOS: self.BOS_IDX, 
            self.EOS: self.EOS_IDX,
        }
        self.idx2word = {}

        self._done = False
        if index_file is not None:
            self.from_file(index_file)
        elif len(extra_keys) > 0:
            self.from_list(extra_keys)
        self._name = name

    @property
    def name(self):
        return self._name
    
    @property
    def word_list(self):
        return [self.idx2word[k] for k in self.idx2word.keys() if k > 3]

    def from_file(self, index_file):
        assert not self._done, "the indexer has already been initialized." 
        with open(index_file, "r") as fr:
            for line in fr:
                line = line.strip().split()
                word, idx = line[0], int(line[1])
                if self.word2idx.get(word, None) is None:
                    assert idx == len(self.word2idx)
                    self.word2idx[word] = idx
        for word, idx in self.word2idx.items():
            self.idx2word[idx] = word
        self._done = True

    def from_list(self, word_list):
        assert not self._done, "the indexer has already been initialized." 
        for _, word in enumerate(word_list):
            if self.word2idx.get(word, None) is None:
                self.word2idx[word] = len(self.word2idx) 
        for word, idx in self.word2idx.items():
            self.idx2word[idx] = word
        self._done = True

    def idx(self, token):
        return self.word2idx.get(token, self.word2idx[self.UNK])  

    def str(self, idx):
        return self.idx2word.get(idx, self.UNK)  

    def has(self, token):
        return token in self.word2idx  

    def add(self, tokens):
        assert isinstance(tokens, list)
        for token in tokens:
            if token in self.word2idx:
                continue
            token_idx = len(self.word2idx)
            self.word2idx[token] = token_idx 
            self.idx2word[token_idx] = token

    def get_special_tokens(self):
        return [self.PAD, self.UNK, self.BOS, self.EOS]

    def get_special_token_masks(self, sequences, special_tokens=None):
        mask = torch.full(sequences.shape, 0, device=sequences.device).bool()
        special_tokens = (
            self.get_special_tokens() if special_tokens is None else special_tokens
        )
        special_token_indice = self(special_tokens)
        for token_idx in special_token_indice:
            sub_mask = sequences == token_idx
            mask = mask | sub_mask
        return mask 

    def __getitem__(self, idx):
        return self.idx(idx)

    def __call__(self, key):
        if isinstance(key, int):
            return self.str(key)
        elif isinstance(key, str):
            return self.idx(key)
        elif isinstance(key, list):
            return [self(k) for k in key] 
        else:
            raise ValueError("type({}) must be `int` or `str`".format(key)) 

    def __len__(self):
        return len(self.word2idx)

def make_vocab(index_file, extra_keys=[], name=""):
    vocab = Indexer(index_file, extra_keys, name)    
    return vocab

def register_indexer(name, index_file, extra_keys=[]):
    """
    Register a pre-defined indexer from a file.

    Args:
        name (str): the name that identifies a indexer, e.g. "co_2017_topk".
        index_file (str): path to the index file.
        extra_keys (dict): directory which contains extra word-index mappings.
    """
    assert isinstance(name, str), name
    #assert isinstance(index_file, (str, os.PathLike)), index_file
    assert isinstance(extra_keys, list), extra_keys
    # 1. register a function which returns dicts
    try:
        DatasetCatalog.register(
            name, 
            lambda: make_vocab(
                index_file, extra_keys=extra_keys, name=name
            )
        )
    except Exception as e:
        warnings.warn(f"{e}")

def mask_tokens(
    sequences, mlm_prob, vocab, train=False, target_words=[], special_token_masks=None, at_least_one=False
):
    inputs = sequences.clone()
    labels = sequences.clone()
    device = sequences.device
    
    if special_token_masks is None:
        special_token_masks = vocab.get_special_token_masks(sequences) 
    else:
        special_token_masks = special_token_masks.bool()

    if not train and len(target_words) > 0:
        #seqs = "\n".join([" ".join(vocab(seq)) for seq in sequences.cpu().tolist()])
        target_masks = vocab.get_special_token_masks(
            sequences, special_tokens=target_words
        )
        inputs[target_masks] = vocab("<mask>")
        labels[~target_masks] = -100 
        return inputs, labels

    if at_least_one:
        B = inputs.shape[0]
        weights = torch.tensor([1, 1, 1], dtype=torch.float, device=device)
        indice = torch.multinomial(weights, B, replacement=True).unsqueeze(1)

        mask = torch.full(sequences.shape, 0, device=device).bool()
        mask = mask & special_token_masks
        mask.scatter_(1, indice + 1, True)

        inputs[mask] = vocab("<mask>")
        labels[~mask] = -100
        return inputs, labels

    prob_matrix = torch.full(sequences.shape, mlm_prob, device=device)
    prob_matrix.masked_fill_(special_token_masks, value=0.)
    all_masked_indice = torch.bernoulli(prob_matrix).bool()
    labels[~all_masked_indice] = -100 # ignore losses from unmasked tokens 
    
    indice_masked_prob = 1.0 #0.8
    indice_masked = torch.bernoulli(torch.full(
        sequences.shape, indice_masked_prob, device=device
    )).bool() & all_masked_indice
    inputs[indice_masked] = vocab("<mask>") 

    indice_random_prob = 0.0 #0.5
    indice_random = torch.bernoulli(torch.full(
        sequences.shape, indice_random_prob, device=device
    )).bool() & all_masked_indice & ~indice_masked
    random_tokens = torch.randint(
        len(vocab), sequences.shape, dtype=torch.long, device=device
    )
    inputs[indice_random] = random_tokens[indice_random]

    return inputs, labels
