"""Single source of truth for ViT model dimensions.

Replaces the per-model hardcoded constants that used to live in
``compiler/graph_extract.py``. One ``ModelConfig`` flows through the IR
builder, compiler, and codegen so the same toolchain can target DeiT-tiny,
ViT-B/16, and (later) any HuggingFace ViTConfig-compatible model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SYS_DIM = 16  # systolic-array tile size; padding multiple for all dims.


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


@dataclass(frozen=True)
class ModelConfig:
    """ViT-family model dimensions.

    ``module_prefix`` controls the state-dict key namespace
    (``vit.encoder.layer.{i}`` for both DeiT-tiny and ViT-B when loaded
    through ``ViTForImageClassification``).
    """

    embed_dim: int
    num_heads: int
    depth: int
    mlp_dim: int
    image_size: int
    patch_size: int
    num_classes: int
    module_prefix: str = "vit"

    def __post_init__(self) -> None:
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @property
    def num_patches(self) -> int:
        side = self.image_size // self.patch_size
        return side * side

    @property
    def seq_len(self) -> int:
        # +1 for the CLS token.
        return self.num_patches + 1

    @property
    def seq_len_pad(self) -> int:
        return _ceil_to_multiple(self.seq_len, SYS_DIM)

    @property
    def patch_dim(self) -> int:
        return 3 * self.patch_size * self.patch_size

    @property
    def pad_rows(self) -> int:
        return self.seq_len_pad - self.seq_len

    @property
    def mlp_ratio(self) -> int:
        # Integer ratio when divisible; otherwise approximate.
        return self.mlp_dim // self.embed_dim

    # ── Named constructors ────────────────────────────────────────────────────

    @classmethod
    def deit_tiny(cls) -> "ModelConfig":
        return cls(
            embed_dim=192,
            num_heads=3,
            depth=12,
            mlp_dim=768,
            image_size=224,
            patch_size=16,
            num_classes=1000,
        )

    @classmethod
    def vit_base(cls) -> "ModelConfig":
        return cls(
            embed_dim=768,
            num_heads=12,
            depth=12,
            mlp_dim=3072,
            image_size=224,
            patch_size=16,
            num_classes=1000,
        )

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> "ModelConfig":
        """Build a ModelConfig from a HuggingFace ``ViTConfig`` (or DeiT, etc.).

        Reads ``hidden_size``, ``num_attention_heads``, ``num_hidden_layers``,
        ``intermediate_size``, ``image_size``, ``patch_size``, ``num_labels``.
        """
        return cls(
            embed_dim=int(hf_config.hidden_size),
            num_heads=int(hf_config.num_attention_heads),
            depth=int(hf_config.num_hidden_layers),
            mlp_dim=int(hf_config.intermediate_size),
            image_size=int(hf_config.image_size),
            patch_size=int(hf_config.patch_size),
            num_classes=int(getattr(hf_config, "num_labels", 1000)),
        )
