"""Model frontends — bridge HuggingFace (or other) models to the TACCEL toolchain.

A frontend produces two things from a user-supplied model:

1. A :class:`taccel.model_config.ModelConfig` capturing the dimensions.
2. A ``state_dict`` mapping the canonical TACCEL parameter names
   (``vit.encoder.layer.{i}.attention.attention.query.weight`` etc.) to
   tensors that the compiler can consume.

The IR graph itself is purely a function of the config — see
``taccel.compiler.graph_extract.extract_vit_graph(cfg)``.
"""
from .hf_vit import load_hf_vit

__all__ = ["load_hf_vit"]
