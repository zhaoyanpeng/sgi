import os
import PIL
import torch
import random
import json, copy
import numpy as np
import itertools
from torch import nn
from nltk import word_tokenize
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torch import nn
from PIL import Image

from .toy import build_dataloader
from .scene_function import *

""" To load images from CLEVR and AbstractScene.
"""

def vgg_transform(resize_size=256, crop_size=224):
    # https://pytorch.org/hub/pytorch_vision_vgg/
#     print(resize_size, crop_size)
    preprocess = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess


class AbstractSceneLoader(torch.utils.data.Dataset):
    def __init__(
        self, cfg, data_file, decoder_vocab, encoder_vocab=None, train=True, cate_vocab=None
    ):
        self.color = (255, 255, 255) 
        droot = cfg.data_root if cfg.data_root == cfg.more_root else cfg.more_root 
        self.vroot = f"{droot}/RenderedScenes"
        self.oroot = f"{cfg.dump_root}"

        self.transform = vgg_transform(resize_size=224) 
        self.components, self.scene_layouts = load_artifacts(droot, seed=1213)

        self.num_image_per_class = 10
        self.length = len(self.scene_layouts) * self.num_image_per_class

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        iimage = index % self.num_image_per_class
        vclass = index // self.num_image_per_class

        image, confs, vname = load_image_from_class(
            self.scene_layouts, self.vroot, vclass, iimage
        )
        names, bboxes, masks, arts = get_name_box_mask_art(confs, self.components)

        _, objects = segment_image_abscene(
            [image], reversed(arts[:]), reversed(masks[:]), reversed(bboxes[:]), 
            n_per_line=1, margin=0, color=self.color, mode="RGBA"
        )
         
        objects = torch.cat([
            self.transform(
                expand2square(obj, self.color).convert("RGB")
            ).unsqueeze(0) for obj in objects 
        ], 0)

        vname = vname.rsplit(".", 1)[0]
        vfile = f"{self.oroot}/{vname}"
        return {"objects": objects, "vfile": vfile}

class ClevrSceneLoader(torch.utils.data.Dataset):
    def __init__(
        self, cfg, data_file, decoder_vocab, encoder_vocab=None, train=True, cate_vocab=None
    ):
        self.color = (255, 255, 255) 
        droot = cfg.data_root if cfg.data_root == cfg.more_root else cfg.more_root 
        self.vroot = f"{droot}/images"
        self.oroot = f"{cfg.dump_root}"

        self.transform = vgg_transform(resize_size=224) 
        with open(data_file, "r") as fr:
            self.dataset = [json.loads(line) for line in fr]
        self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        scene = self.dataset[index]
        _, objects, vname = segment_image_clevr(
            scene, self.vroot, n_per_line=1, margin=0, color=self.color, mode="RGBA"
        )

        objects = torch.cat([
            self.transform(
                expand2square(obj, self.color).convert("RGB")
            ).unsqueeze(0) for obj in objects 
        ], 0)
        
        vname = vname.rsplit(".", 1)[0]
        vfile = f"{self.oroot}/{vname}"
        return {"objects": objects, "vfile": vfile}

def build_scene_dataset(cfg, train, echo):
    ifile = cfg.data_root + cfg.data_name
    #assert os.path.isfile(ifile), f"not a data file {ifile}"
    evalloader = testloader = encoder_vocab = decoder_vocab = cate_vocab = None
    cls = ClevrSceneLoader if cfg.name == "clevr" else AbstractSceneLoader
    dataloader = build_dataloader(
        cfg, cls, ifile, encoder_vocab, decoder_vocab, cate_vocab, train, echo, msg="main"
    )
    return dataloader, evalloader, testloader, encoder_vocab, decoder_vocab, cate_vocab
