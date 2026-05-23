"""HuggingFace ViT-family loader.

``transformers.ViTForImageClassification`` exposes both DeiT-tiny and ViT-B/16
under a ``vit.*`` state_dict prefix, which matches the parameter names the
TACCEL compiler emits. So the "frontend" for these models is essentially a
config-and-state-dict pass-through.

Models known to load cleanly here:
  - facebook/deit-tiny-patch16-224
  - google/vit-base-patch16-224

Other ViT variants will load if their HuggingFace config exposes the standard
``hidden_size`` / ``num_attention_heads`` / ``num_hidden_layers`` /
``intermediate_size`` / ``image_size`` / ``patch_size`` fields and the
state_dict key naming matches the ``vit.*`` convention.
"""
from __future__ import annotations

from typing import Dict, Tuple

from ..model_config import ModelConfig


def load_hf_vit(model_name: str, *, local_files_only: bool = False) -> Tuple[ModelConfig, Dict]:
    """Load a HuggingFace ViT-family model.

    Returns ``(cfg, state_dict)`` where ``cfg`` is the derived
    :class:`ModelConfig` and ``state_dict`` is the model's
    parameter dictionary (PyTorch tensors).
    """
    # Imported lazily so the rest of the package doesn't drag in torch/HF.
    from transformers import ViTForImageClassification

    model = ViTForImageClassification.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    cfg = ModelConfig.from_hf_config(model.config)
    return cfg, model.state_dict()
