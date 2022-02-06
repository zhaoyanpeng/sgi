from .helper import * 
from .sgi import SGI 
from .mini_tf import MiniTFLM

from fvcore.common.registry import Registry

SGI_MODELS_REGISTRY = Registry("SGI_MODELS")
SGI_MODELS_REGISTRY.__doc__ = """
Registry for parser models.
"""

def build_main_model(cfg, echo):
    return SGI_MODELS_REGISTRY.get(cfg.worker)(cfg, echo)

SGI_MODELS_REGISTRY.register(SGI)
SGI_MODELS_REGISTRY.register(MiniTFLM)
