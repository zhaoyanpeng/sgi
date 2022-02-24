from omegaconf import OmegaConf
import os, re
import torch
from collections import OrderedDict

__all__ = ["load_checkpoint"]

def load_checkpoint(cfg, echo):
    model_file = f"{cfg.model_root}/{cfg.model_name}/{cfg.model_file}"
    try:
        checkpoint = torch.load(model_file, map_location="cpu")
        echo(f"Loading from {model_file}")
    except Exception as e:
        echo(f"Failed to load the checkpoint `{model_file}` {e}")
        return (None,) * 6
    local_cfg = checkpoint["cfg"]
    local_str = OmegaConf.to_yaml(local_cfg)
    if cfg.verbose:
        echo(f"Old configs:\n\n{local_str}")
    nmodule = len(checkpoint["model"])
    if nmodule == 2:
        encoder_head_sd, decoder_head_sd = checkpoint["model"]
        return local_cfg, None, None, encoder_head_sd, decoder_head_sd, None
    elif nmodule == 3:
        encoder_head_sd, decoder_head_sd, loss_head_sd = checkpoint["model"]
        return local_cfg, None, None, encoder_head_sd, decoder_head_sd, loss_head_sd
    elif nmodule == 5:
        backbone_head_sd, relation_head_sd, encoder_head_sd, decoder_head_sd, loss_head_sd = checkpoint["model"]
        return local_cfg, backbone_head_sd, relation_head_sd, encoder_head_sd, decoder_head_sd, loss_head_sd
    elif nmodule == 6:
        backbone_head_sd, relation_head_sd, encoder_head_sd, decoder_head_sd, loss_head_sd, vi_head_sd = checkpoint["model"]
        return local_cfg, backbone_head_sd, relation_head_sd, encoder_head_sd, decoder_head_sd, loss_head_sd, vi_head_sd
    else:
        raise ValueError(f"I don't know how to parse the checkpoint: # module is {nmodule}.")

