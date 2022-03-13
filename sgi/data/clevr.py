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

def compute_all_relationships(scene_struct, objects, eps=0.2):
  """
  Computes relationships between all pairs of objects in the scene.

  Returns a dictionary mapping string relationship names to lists of lists of
  integers, where output[rel][i] gives a list of object indices that have the
  relationship rel with object i. For example if j is in output['left'][i] then
  object j is left of object i.

  https://github.com/facebookresearch/clevr-dataset-gen/blob/f0ce2c81750bfae09b5bf94d009f42e055f2cb3a/image_generation/render_images.py#L448
  """
  all_relationships = {}
  for name, direction_vec in scene_struct['directions'].items():
    if name == 'above' or name == 'below': continue
    all_relationships[name] = []
    for i, obj1 in enumerate(objects): #scene_struct['objects']):
      coords1 = obj1['3d_coords']
      related = set()
      for j, obj2 in enumerate(objects): #scene_struct['objects']):
        if obj1 == obj2: continue
        coords2 = obj2['3d_coords']
        diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
        dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
        if dot > eps:
          related.add(j)
      all_relationships[name].append(sorted(list(related)))
  return all_relationships

class ClevrDataLoader(torch.utils.data.Dataset):
    UNK = "<unk>"
    DUM = "<dum>"
    image_size = (320, 480) # (h, w) 
    def __init__(
        self, cfg, data_file, decoder_vocab, encoder_vocab=None, train=True, cate_vocab=None
    ):
        self.decoder_vocab = decoder_vocab
        self.encoder_vocab = encoder_vocab
        self.cate_vocab = cate_vocab
        self.train = train

        meta = MetadataCatalog.get("clevr")
        unk_idx = meta.thing_classes.index(self.UNK)
        dum_idx = cate_vocab(self.DUM) if cate_vocab is not None else None
        cate_type = cate_vocab.name if cate_vocab is not None else None
        self.caption_key = "caption" if self.cate_vocab is None else "new_caption" 
        self.cate_max_len = cfg.cate_max_len

        def process_fields(item):
            obj_cates = list()
            obj_names = list()
            obj_boxes = list()
            obj_classes = list()

            #sorted_annos = sorted(item["annotations"], key=lambda x: x["obj_id"])
            sorted_annos = sorted(item["annotations"], key=lambda x: x["bbox"][0])
            item["relationships"] = compute_all_relationships(item, sorted_annos)

            for anno in sorted_annos:
                x, y, w, h = anno["bbox"]
                obj_boxes.append([
                      x / self.image_size[1], y / self.image_size[0],
                      (x + w) / self.image_size[1], (y + h) / self.image_size[0]
                ])
                category = anno["category"]
                if cate_type == "atomic_object":
                    obj_cates.append(cate_vocab(category))
                elif cate_type == "atomic_triplet":
                    obj_cates.append(category)
                elif cate_type is not None and "sample_word" in cate_type:
                    obj_cates.append(word_tokenize(category.lower()))
                obj_names.append(encoder_vocab(word_tokenize(category.lower())))
                obj_classes.append(
                    meta.thing_classes.index(category) 
                    if category in meta.thing_classes else unk_idx 
                )
            if cfg.add_dummy:
                x, y, w, h = 0, 0, self.image_size[1], self.image_size[0]
                obj_boxes.insert(0, [
                      x / self.image_size[1], y / self.image_size[0],
                      (x + w) / self.image_size[1], (y + h) / self.image_size[0]
                ])
                if cate_type == "atomic_object":
                    obj_cates.insert(0, dum_idx)
                elif cate_type == "atomic_triplet":
                    obj_cates.insert(0, self.DUM)
                elif cate_type is not None and "sample_word" in cate_type:
                    obj_cates.insert(0, [self.DUM])
                obj_names.insert(0, [encoder_vocab.BOS_IDX] * 4) # always 4-word category
                obj_classes.insert(0, unk_idx)

            obj_idxes = list(range(len(obj_classes))) # why needed? 

            new_caption = list()
            if cate_type is not None:
                relations = item["relationships"]
                for idx, cate in enumerate(obj_cates):
                    if cfg.add_dummy and idx == 0:
                        continue
                    for rel in list(cfg.relation_words): #["left", "right", "front", "behind"]:
                        if cate_type == "atomic_object": # list of [int, int, int]
                            rel_idx = cate_vocab(rel)
                            for x_rel_of_cate in relations[rel][idx]:
                                token = [obj_cates[x_rel_of_cate], rel_idx, cate]
                                #token = [obj_cates[x_rel_of_cate], cate, rel_idx]
                                new_caption.append(token) 
                        elif cate_type == "atomic_triplet": # list of [str] 
                            for x_rel_of_cate in relations[rel][idx]:
                                token = [" ".join([obj_cates[x_rel_of_cate], cate, rel])]
                                new_caption.append(cate_vocab(token))
                        elif "_oor" in cate_type: # list of [1-word str's]
                            for x_rel_of_cate in relations[rel][idx]:
                                token = obj_cates[x_rel_of_cate] + cate + [rel]
                                new_caption.append(token)
                        elif "_oro" in cate_type: # list of [1-word str's]
                            for x_rel_of_cate in relations[rel][idx]:
                                token = obj_cates[x_rel_of_cate] + [rel] + cate
                                new_caption.append(token)
                        else:
                            raise ValueError(f"Bad cate_vocab.name = {cate_vocab.name}")
            return obj_idxes, obj_names, obj_boxes, obj_classes, [new_caption]

        def manipulate_relations(item):
            nobj = len(item["annotations"])
            for _, rel_list in item["relationships"].items():
                for _, rels in enumerate(rel_list):
                    for idx, _ in enumerate(rels):
                        rels[idx] += 1
                    rels.insert(0, 0)
                rel_list.insert(0, list(range(1, nobj + 1)))

        def filter_captions(captions):
            relations = list(cfg.relation_words)
            if cate_type is not None or len(relations) == 0:
                return captions
            #print("\n".join([" ".join(x) for x in captions]))
            final = list()
            for caption in captions:
                for relation in relations:
                    if relation in caption:
                        final.append(caption)
                        break
            #print("\n".join([" ".join(x) for x in final]))
            return final

        max_len = -1

        self.dataset = []
        with open(data_file, "r") as fr:
            for line in fr:
                item = json.loads(line)
                #max_len = max(max_len, max([len(x) for x in item["captions"]]))
                captions = filter_captions(item["captions"])
                if len(captions) == 0:
                    continue
                if cfg.add_dummy:
                    manipulate_relations(item)
                new_item = {
                    "relations": item["relationships"],
                    "file_name": item["file_name"], 
                    "caption": captions,
                }
                (
                    new_item["obj_idxes"], 
                    new_item["obj_names"], 
                    new_item["obj_boxes"], 
                    new_item["obj_classes"], 
                    new_item["new_caption"]
                ) = process_fields(item)

                if len(new_item["obj_classes"]) != len(set(new_item["obj_classes"])):
                    continue # ambiguity

                self.dataset.append(new_item)
        self.length = len(self.dataset)

    def _shuffle(self):
        indice = torch.randperm(self.length).tolist() 
        indice = sorted(indice, key=lambda k: self.dataset[k]["obj_idxes"].size(0))
        self.dataset = [self.dataset[k] for k in indice]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        item = self.dataset[index]
        captions = item[self.caption_key]
        icaption = np.random.choice(len(captions), 1)[0] if self.train else 0 
        caption = captions[icaption]

        sample = {
            "width": self.image_size[1],
            "height": self.image_size[0], 
            "obj_classes": item["obj_classes"],
            "relations": item["relations"], 
            "file_name": item["file_name"], 
            "obj_idxes": item["obj_idxes"],
            "obj_names": item["obj_names"],
            "obj_boxes": item["obj_boxes"],
        }

        if self.cate_vocab is None:
            sample["caption"] = self.decoder_vocab(
                [self.decoder_vocab.BOS] + caption + [self.decoder_vocab.EOS]
            )
        else: 
            triplets = caption 
            ntriplet = len(caption)
            if ntriplet > self.cate_max_len:
                if True or self.train:
                    indice = np.random.choice(
                        ntriplet, self.cate_max_len, replace=False
                    )
                    #indice.sort()
                    triplets = [caption[i] for i in indice]
                else: # deterministic test samples
                    triplets = caption[:self.cate_max_len]
            if "_oor" in self.cate_vocab.name or "_oro" in self.cate_vocab.name:
                triplets = [self.cate_vocab(triplet) for triplet in triplets]
            triplets = [[self.cate_vocab.BOS_IDX]] + triplets + [[self.cate_vocab.EOS_IDX]]
            sample["caption"] = list(itertools.chain.from_iterable(triplets))
        return sample 

def collate_fun(data):
    union = {
        k: [item.get(k) for item in data] for k in set().union(*data)
    }
    return union 

def register_clevr_metadata(name="clevr"):
    attribute_words = """
        {"size": ["small", "large"], "color": ["gray", "blue", "brown", "yellow", "red", "green", "purple", "cyan"],
        "material": ["rubber", "metal"], "shape": ["cube", "sphere", "cylinder"]}
    """
    attributes = []
    attribute_words = json.loads(attribute_words)
    for size in attribute_words["size"]:
        for shape in attribute_words["shape"]:
            for color in attribute_words["color"]:
                for material in attribute_words["material"]:
                    attributes.append(f"{size} {shape} {color} {material}")
    attributes.append("<unk>")
    nattribute = len(attributes)
    thing_dataset_id_to_contiguous_id = {i: i for i in range(nattribute)}
    thing_classes = [x for x in attributes] 
    thing_colors = ["" for x in range(nattribute)]
    metadata = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    MetadataCatalog.get(name).set(**metadata)

def connect_class2token(vocab, meta):
    classes = meta.thing_classes
    token2class = dict()
    for iclass, name in enumerate(classes):
        if not vocab.has(name):
            assert False, f"{name} is not in vocab."
        token2class[vocab(name)] = iclass
    return token2class

def process_boxes(obj_boxes, add_dummy=False, dummy_pos_type="min"):
    """ create pos for the dummy object.
    """
    if not add_dummy:
        return obj_boxes
    if dummy_pos_type == "min":
        obj_boxes[:, 0] = 0
    elif dummy_pos_type == "max":
        obj_boxes[:, 0, 0] = obj_boxes[:, 0, 2]
        obj_boxes[:, 0, 1] = obj_boxes[:, 0, 3]
    elif dummy_pos_type == "mid":
        obj_boxes[:, 0, 0] = (obj_boxes[:, 0, 0] + obj_boxes[:, 0, 2]) / 2
        obj_boxes[:, 0, 1] = (obj_boxes[:, 0, 1] + obj_boxes[:, 0, 3]) / 2
        obj_boxes[:, 0, 2] = obj_boxes[:, 0, 3] = 0
    elif dummy_pos_type == "cx":
        obj_boxes[:, 0, 0] = (obj_boxes[:, 0, 0] + obj_boxes[:, 0, 2]) / 2
        obj_boxes[:, 0, 1] = obj_boxes[:, 0, 2] = obj_boxes[:, 0, 3] = 0
    return obj_boxes

def process_clevr_batch(union, encoder_vocab, decoder_vocab, cate_vocab, device, max_num_obj=26, **kwargs):
    sequences = np.array(list(itertools.zip_longest(
        *union["caption"], fillvalue=decoder_vocab.PAD_IDX
    ))).T
    obj_idxes = np.array(list(itertools.zip_longest(
        *union["obj_idxes"], fillvalue=max_num_obj
    ))).T
    obj_boxes = np.array(list(itertools.zip_longest(
        *union["obj_boxes"], fillvalue=[0.] * 4
    ))).transpose(1, 0, 2)
    obj_boxes = process_boxes(obj_boxes).astype(np.float32)
    
    obj_names = union["obj_names"]
    ntoken = max([max([len(name) for name in sample]) for sample in obj_names])
    obj_names = [
        [name + [encoder_vocab.PAD_IDX] * (ntoken - len(name)) for name in sample] 
        for sample in obj_names
    ]
    obj_names = np.array(list(itertools.zip_longest(
        *union["obj_names"], fillvalue=[encoder_vocab.PAD_IDX] * ntoken
    ))).transpose(1, 0, 2)

    width = torch.tensor(union["width"]).unsqueeze(-1).unsqueeze(-1).to(device)
    height = torch.tensor(union["height"]).unsqueeze(-1).unsqueeze(-1).to(device)

    obj_name_lengths = (obj_names != encoder_vocab.PAD_IDX).sum(-1).clip(min=1)
    obj_masks = obj_idxes == max_num_obj # masked out if true 

    items = [torch.tensor(x, device=device) for x in [
        obj_idxes, obj_boxes, obj_masks, sequences, obj_names, obj_name_lengths
    ]]
    items = tuple(items) + ((height, width),)
    return items 

def build_clevr_dataset(cfg, train, echo):
    # file based vocab
    dec_vocab_file = cfg.data_root + cfg.dec_vocab_name
    try:
        register_indexer(cfg.dec_vocab_name, dec_vocab_file)
    except Exception as e:
        echo(f"{e}")
    decoder_vocab = DatasetCatalog.get(cfg.dec_vocab_name) 

    enc_vocab_file = cfg.data_root + cfg.enc_vocab_name
    if os.path.isfile(enc_vocab_file):
        register_indexer(cfg.enc_vocab_name, enc_vocab_file)
        encoder_vocab = DatasetCatalog.get(cfg.enc_vocab_name) 
    else:
        encoder_vocab = decoder_vocab

    # customized vocab
    register_clevr_metadata("clevr")
    # list based vocab
    meta = MetadataCatalog.get("clevr")
    word_list = copy.deepcopy(meta.thing_classes)
    rels = ["left", "right", "front", "behind"]
    cate_vocab_name = cfg.cate_type
    if cate_vocab_name == "atomic_object":
        if cfg.add_dummy:
            rels.append("<dum>")
        word_list.extend(rels)
        register_indexer(cate_vocab_name, None, extra_keys=word_list)
        cate_vocab = DatasetCatalog.get(cate_vocab_name) 
        pass # (color, size, material, shape) as a whole
    elif cate_vocab_name == "atomic_triplet":
        new_word_list = list()
        word_list.append("<unk>")
        for i, w0 in enumerate(word_list):
            for j, w1 in enumerate(word_list):
                for k, rel in enumerate(rels):
                    word = " ".join([w0, w1, rel]) 
                    new_word_list.append(word)
        if cfg.add_dummy:
            for i, w0 in enumerate(word_list):
                for k, rel in enumerate(rels):
                    word = " ".join(["<dum>", w0, rel]) 
                    new_word_list.append(word)
                    word = " ".join([w0, "<dum>", rel]) 
                    new_word_list.append(word)
        register_indexer(cate_vocab_name, None, extra_keys=new_word_list)
        cate_vocab = DatasetCatalog.get(cate_vocab_name) 
        pass # (obj0, obj1, rel) as a whole
    elif "sample_word" in cate_vocab_name:
        word_list = decoder_vocab.word_list
        if cfg.add_dummy:
            word_list.append("<dum>")
        register_indexer(cate_vocab_name, None, extra_keys=word_list)
        cate_vocab = DatasetCatalog.get(cate_vocab_name) 
        pass # do 
    else:
        cate_vocab = None
        pass # do 

    decoder_vocab = cate_vocab if cate_vocab is not None else decoder_vocab
    if cfg.mlm_prob > 0.:
        decoder_vocab.add(["<mask>"])
        echo("Add `<mask>' for MLM training.")

    dataloader = evalloader = testloader = None

    # train
    ifile = cfg.data_root + (cfg.data_name if train else cfg.eval_name) 
    assert os.path.isfile(ifile), f"not a data file {ifile}"
    dataloader = build_dataloader(
        cfg, ClevrDataLoader, ifile, encoder_vocab, decoder_vocab, cate_vocab, train, echo, msg="main"
    )
    # eval
    ifile = cfg.data_root + cfg.eval_name if train else "IGNORE_ME"
    if os.path.isfile(ifile):
        evalloader = build_dataloader(
            cfg, ClevrDataLoader, ifile, encoder_vocab, decoder_vocab, cate_vocab, False, echo, msg="eval"
        )
    # test
    ifile = cfg.data_root + cfg.test_name if train else "IGNORE_ME"
    if os.path.isfile(ifile):
        testloader = build_dataloader(
            cfg, ClevrDataLoader, ifile, encoder_vocab, decoder_vocab, cate_vocab, False, echo, msg="test"
        )
    return dataloader, evalloader, testloader, encoder_vocab, decoder_vocab, cate_vocab
