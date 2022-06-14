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

class CopyDataLoader(torch.utils.data.Dataset):
    UNK = "<unk>"
    DUM = "<dum>"
    image_size = (1, 1) # (h, w) 
    def __init__(
        self, cfg, data_file, decoder_vocab, encoder_vocab=None, train=True, cate_vocab=None
    ):
        meta = MetadataCatalog.get("copy")
        unk_idx = meta.thing_classes.index(self.UNK)
        dum_idx = cate_vocab(self.DUM) if cate_vocab is not None else None
        self.dataset = []
        with open(data_file, "r") as fr:
            for line in fr:
                caption = json.loads(line)
                new_item = {
                    "relations": [], "new_caption": [], "file_name": "", "height": 1, "width": 1,
                    "caption": decoder_vocab([decoder_vocab.BOS] + caption + [decoder_vocab.EOS]),
                    "obj_idxes": list(range(len(caption))),
                    "obj_names": encoder_vocab(caption),
                    "obj_boxes": list(range(len(caption))),
                    "obj_classes": [meta.thing_classes.index(category)
                        if category in meta.thing_classes else unk_idx for category in caption
                    ]
                }
                self.dataset.append(new_item)
                if not train:
                    pass #break
        self.length = len(self.dataset)

    def _shuffle(self):
        indice = torch.randperm(self.length).tolist() 
        indice = sorted(indice, key=lambda k: len(self.dataset[k]["obj_idxes"]))
        self.dataset = [self.dataset[k] for k in indice]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.dataset[index]

def collate_fun(data):
    union = {
        k: [item.get(k) for item in data] for k in set().union(*data)
    }
    return union 

def register_copy_metadata(name="copy"):
    chars = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
        'v', 'w', 'x', 'y', 'z', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an', 'ao',
        'ap', 'aq', 'ar', 'as', 'at', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'ba', 'bc', 'bd', 'be', 'bf', 'bg', 'bh',
        'bi', 'bj', 'bk', 'bl', 'bm', 'bn', 'bo', 'bp', 'bq', 'br', 'bs', 'bt', 'bu', 'bv', 'bw', 'bx', 'by', 'bz',
        'ca', 'cb', 'cd', 'ce', 'cf', 'cg', 'ch', 'ci', 'cj', 'ck', 'cl', 'cm', 'cn', 'co', 'cp', 'cq', 'cr', 'cs',
        'ct', 'cu', 'cv', 'cw', 'cx', 'cy', 'cz', 'da', 'db', 'dc', 'de', 'df', 'dg', 'dh', 'di', 'dj', 'dk', 'dl',
        'dm', 'dn', 'do', 'dp', 'dq', 'dr', 'ds', 'dt', 'du', 'dv', 'dw', 'dx', 'dy', 'dz', 'ea', 'eb', 'ec', 'ed',
        'ef', 'eg', 'eh', 'ei', 'ej', 'ek', 'el', 'em', 'en', 'eo', 'ep', 'eq', 'er', 'es', 'et', 'eu', 'ev', 'ew',
        'ex', 'ey', 'ez', 'fa', 'fb', 'fc', 'fd', 'fe', 'fg', 'fh', 'fi', 'fj', 'fk', 'fl', 'fm', 'fn', 'fo', 'fp',
        'fq', 'fr', 'fs', 'ft', 'fu', 'fv', 'fw', 'fx', 'fy', 'fz', 'ga', 'gb', 'gc', 'gd', 'ge', 'gf', 'gh', 'gi',
        'gj', 'gk', 'gl', 'gm', 'gn', 'go', 'gp', 'gq', 'gr', 'gs', 'gt', 'gu', 'gv', 'gw', 'gx', 'gy', 'gz', 'ha',
        'hb', 'hc', 'hd', 'he', 'hf', 'hg', 'hi', 'hj', 'hk', 'hl', 'hm', 'hn', 'ho', 'hp', 'hq', 'hr', 'hs', 'ht',
        'hu', 'hv', 'hw', 'hx', 'hy', 'hz', 'ia', 'ib', 'ic', 'id', 'ie', 'if', 'ig', 'ih', 'ij', 'ik', 'il', 'im',
        'in', 'io', 'ip', 'iq', 'ir', 'is', 'it', 'iu', 'iv', 'iw', 'ix', 'iy', 'iz', 'ja', 'jb', 'jc', 'jd', 'je',
    ]
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

def process_copy_batch(union, encoder_vocab, decoder_vocab, cate_vocab, device, max_num_obj=512, **kwargs):
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
    return items 

def build_dataloader(
    cfg, data_loader_cls, ifile, encoder_vocab, decoder_vocab, cate_vocab, train, echo, msg="", **kwargs
):
    dataset = data_loader_cls(
        cfg, ifile, 
        encoder_vocab=encoder_vocab, 
        decoder_vocab=decoder_vocab, 
        cate_vocab=cate_vocab,
        train=train,
        **kwargs,
    )
    if not train: 
        sampler = torch.utils.data.SequentialSampler(dataset)
    else:
        sampler = torch.utils.data.RandomSampler(dataset) 
        #sampler = SortedSequentialSampler(dataset) 
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=cfg.batch_size, 
        #num_workers=cfg.num_proc,
        shuffle=False,
        sampler=sampler,
        pin_memory=True, 
        collate_fn=collate_fun
    )
    echo(f"Loading from {ifile} {len(data_loader)} ({len(dataset)}) batches ({msg}).")
    return data_loader

def build_copy_dataset(cfg, train, echo):
    # file based vocab
    dec_vocab_file = cfg.data_root + cfg.dec_vocab_name
    register_indexer(cfg.dec_vocab_name, dec_vocab_file)
    decoder_vocab = DatasetCatalog.get(cfg.dec_vocab_name) 

    enc_vocab_file = cfg.data_root + cfg.enc_vocab_name
    if os.path.isfile(enc_vocab_file):
        register_indexer(cfg.enc_vocab_name, enc_vocab_file)
        encoder_vocab = DatasetCatalog.get(cfg.enc_vocab_name) 
    else:
        encoder_vocab = decoder_vocab
    cate_vocab = None # placeholder

    # customized vocab
    register_copy_metadata("copy")

    dataloader = evalloader = testloader = None

    # train
    ifile = cfg.data_root + cfg.data_name if train else cfg.eval_name 
    assert os.path.isfile(ifile), f"not a data file {ifile}"
    dataloader = build_dataloader(
        cfg, CopyDataLoader, ifile, encoder_vocab, decoder_vocab, cate_vocab, train, echo, msg="main"
    )
    # eval
    ifile = cfg.data_root + cfg.eval_name if train else "IGNORE_ME"
    if os.path.isfile(ifile):
        evalloader = build_dataloader(
            cfg, CopyDataLoader, ifile, encoder_vocab, decoder_vocab, cate_vocab, False, echo, msg="eval"
        )
    # test
    ifile = cfg.data_root + cfg.test_name if train else "IGNORE_ME"
    if os.path.isfile(ifile):
        testloader = build_dataloader(
            cfg, CopyDataLoader, ifile, encoder_vocab, decoder_vocab, cate_vocab, False, echo, msg="test"
        )
    return dataloader, evalloader, testloader, encoder_vocab, decoder_vocab, cate_vocab
