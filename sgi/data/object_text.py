import os
import torch
import random
import json, copy
import numpy as np
import itertools
from torch import nn
from nltk import word_tokenize
from torch.nn.utils.rnn import pad_sequence

from . import MetadataCatalog, DatasetCatalog, SortedSequentialSampler, register_indexer
from ..module import PositionalEncoder
from .toy import build_dataloader

class AbsceneObjectTextLoader(torch.utils.data.Dataset):
    """ Pre-encoded object features and image captions. 
    """
    def __init__(
        self, cfg, data_file, decoder_vocab, encoder_vocab=None, train=True, cate_vocab=None, chunk=[0, 10000] #slice(0, None)
    ):
        # the dataset is provided as a whole, we have to split it into train, val, and test sets.
        self.decoder_vocab = decoder_vocab
        self.encoder_vocab = encoder_vocab
        self.max_length = cfg.max_length
        self.train = train

        droot = cfg.data_root 
        self.vroot = f"{cfg.more_root}"
        
        captions = list()
        with open(data_file, "r") as fr:
            for iline, line in enumerate(fr):
                if iline < chunk[0]:
                    continue
                if iline >= (chunk[0] + chunk[1]):
                    break
                txt = word_tokenize(line.strip().lower())
                captions.append([(txt, len(txt))])
        self.captions = captions
        
        self.num_image_per_class = 10
        max_num_cluster = len(self.captions)
        self.length = max_num_cluster * self.num_image_per_class

        num_cluster = chunk[1] if chunk[1] < max_num_cluster else max_num_cluster

        # build image index in order to load data by class
        indice = [range(self.num_image_per_class)] * num_cluster
        self.indice = [(chunk[0] + i, xx) for i, x in enumerate(indice) for xx in x]
        assert self.length == len(self.indice)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        vclass, iimage = self.indice[index]

        npz_name = f"Scene{vclass}_{iimage}" 
        npz_file = f"{self.vroot}/{npz_name}.npz"
        objects = np.load(npz_file)["v"][1:] # remove background
        num_obj = objects.shape[0]

        vclass = index // self.num_image_per_class
        caption, nword = random.choice(self.captions[vclass])
        if nword > self.max_length:
            start = random.choice(range(nword - self.max_length)) if self.train else 0
            caption = caption[start : start + self.max_length]
        caption = self.decoder_vocab(
            [self.decoder_vocab.BOS] + caption + [self.decoder_vocab.EOS]
        )

        sample = {
            "relations": [], "new_caption": [], "file_name": "", "height": 1, "width": 1,
            "caption": caption,
            "objects": objects,
            "obj_idxes": list(range(num_obj)),
            "obj_names": list(range(num_obj)),
            "obj_boxes": list(range(num_obj)),
            "obj_classes": list(range(num_obj)) 
        }
        return sample 

def register_abscene_metadata(name="abscene"):
    chars = [str(i) for i in range(50)]
    chars.append("<unk>")
    nchar = len(chars)
    thing_dataset_id_to_contiguous_id = {i: i for i in range(nchar)}
    thing_classes = [x for x in chars]
    thing_colors = ["" for x in range(nchar)]
    metadata = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    MetadataCatalog.get(name).set(**metadata)

def process_abscene_batch(union, encoder_vocab, decoder_vocab, cate_vocab, device, max_num_obj=512, **kwargs):
    sequences = np.array(list(itertools.zip_longest(
        *union["caption"], fillvalue=decoder_vocab.PAD_IDX
    ))).T
    obj_idxes = np.array(list(itertools.zip_longest(
        *union["obj_idxes"], fillvalue=max_num_obj
    ))).T
    obj_names = np.array(list(itertools.zip_longest(
        *union["obj_names"], fillvalue=encoder_vocab.PAD_IDX
    ))).T
    obj_boxes = np.array(list(itertools.zip_longest(
        *union["obj_boxes"], fillvalue=0
    ))).T

    obj_names = obj_names[..., None] # add a new axis

    width = torch.tensor(union["width"]).unsqueeze(-1).unsqueeze(-1).to(device)
    height = torch.tensor(union["height"]).unsqueeze(-1).unsqueeze(-1).to(device)

    obj_name_lengths = (obj_names != encoder_vocab.PAD_IDX).sum(-1).clip(min=1)
    obj_masks = obj_idxes == max_num_obj # masked out if true 

    items = [torch.tensor(x, device=device) for x in [
        obj_idxes, obj_boxes, obj_masks, sequences, obj_names, obj_name_lengths
    ]]
    items = tuple(items) + ((height, width),)

    dim = union["objects"][0].shape[-1]
    objects = np.array(list(itertools.zip_longest(
        *union["objects"], fillvalue=np.zeros(dim, dtype=np.float32)
    ))).transpose(1, 0, 2)
    #objects = objects[:, 1:] # remove background
    items += (torch.tensor(objects, device=device),)
    return items 

def build_abscene_dataset(cfg, train, echo):
    # file based vocab
    dec_vocab_file = f"{cfg.data_root}/{cfg.dec_vocab_name}"
    try:
        register_indexer(cfg.dec_vocab_name, dec_vocab_file)
    except Exception as e:
        echo(f"{e}")
    decoder_vocab = DatasetCatalog.get(cfg.dec_vocab_name) 

    enc_vocab_file = f"{cfg.data_root}/{cfg.enc_vocab_name}"
    if os.path.isfile(enc_vocab_file):
        register_indexer(cfg.enc_vocab_name, enc_vocab_file)
        encoder_vocab = DatasetCatalog.get(cfg.enc_vocab_name) 
    else:
        encoder_vocab = decoder_vocab

    # customized vocab
    register_abscene_metadata("abscene")
    cate_vocab = None

    decoder_vocab = cate_vocab if cate_vocab is not None else decoder_vocab
    if cfg.mlm_prob > 0.:
        decoder_vocab.add(["<mask>"])
        echo("Add `<mask>' for MLM training.")

    dataloader = evalloader = testloader = None

    # train
    ifile = f"{cfg.data_root}/Sentences_1002.txt"
    assert os.path.isfile(ifile), f"not a data file {ifile}"
    chunk = cfg.data_name if train else cfg.eval_name
    chunk = list(chunk)
    dataloader = build_dataloader(
        cfg, AbsceneObjectTextLoader, ifile, encoder_vocab, decoder_vocab, cate_vocab, train, echo, chunk=chunk, msg="main"
    )
    # eval
    chunk = cfg.eval_name if train else None 
    if chunk is not None:
        chunk = list(chunk)
        evalloader = build_dataloader(
            cfg, AbsceneObjectTextLoader, ifile, encoder_vocab, decoder_vocab, cate_vocab, False, echo, chunk=chunk, msg="eval"
        )
    # test
    chunk = cfg.test_name if train else None 
    if chunk is not None:
        chunk = list(chunk)
        testloader = build_dataloader(
            cfg, AbsceneObjectTextLoader, ifile, encoder_vocab, decoder_vocab, cate_vocab, False, echo, chunk=chunk, msg="test"
        )
    return dataloader, evalloader, testloader, encoder_vocab, decoder_vocab, cate_vocab
