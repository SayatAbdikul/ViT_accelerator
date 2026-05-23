"""Invariants for ModelConfig — derived fields must agree with the legacy
hardcoded constants for DeiT-tiny, and ViT-B/16 must match HF's config."""
from __future__ import annotations

import pytest

from taccel.model_config import ModelConfig, SYS_DIM


def test_deit_tiny_matches_legacy_constants():
    cfg = ModelConfig.deit_tiny()
    # These mirror the constants that used to live at the top of
    # taccel/compiler/graph_extract.py.
    assert cfg.embed_dim == 192
    assert cfg.depth == 12
    assert cfg.num_heads == 3
    assert cfg.head_dim == 64
    assert cfg.mlp_ratio == 4
    assert cfg.mlp_dim == 768
    assert cfg.seq_len == 197
    assert cfg.patch_size == 16
    assert cfg.image_size == 224
    assert cfg.num_patches == 196
    assert cfg.patch_dim == 768  # 3 × 16 × 16
    assert cfg.num_classes == 1000
    assert cfg.seq_len_pad == 208
    assert cfg.pad_rows == 11


def test_vit_base_dimensions():
    cfg = ModelConfig.vit_base()
    assert cfg.embed_dim == 768
    assert cfg.num_heads == 12
    assert cfg.head_dim == 64
    assert cfg.depth == 12
    assert cfg.mlp_dim == 3072
    assert cfg.mlp_ratio == 4
    assert cfg.seq_len == 197
    assert cfg.seq_len_pad == 208
    # Same patch grid as DeiT-T (224/16=14, +1 CLS).


def test_invalid_dim_combinations_rejected():
    with pytest.raises(ValueError):
        ModelConfig(
            embed_dim=193,  # not divisible by num_heads=3
            num_heads=3,
            depth=12,
            mlp_dim=768,
            image_size=224,
            patch_size=16,
            num_classes=1000,
        )
    with pytest.raises(ValueError):
        ModelConfig(
            embed_dim=192,
            num_heads=3,
            depth=12,
            mlp_dim=768,
            image_size=225,  # not divisible by patch_size=16
            patch_size=16,
            num_classes=1000,
        )


def test_seq_len_pad_is_multiple_of_sys_dim():
    for cfg in [ModelConfig.deit_tiny(), ModelConfig.vit_base()]:
        assert cfg.seq_len_pad % SYS_DIM == 0
        assert cfg.seq_len_pad >= cfg.seq_len


def test_from_hf_config_mirrors_vit_base():
    """ViTConfig fields → ModelConfig should match the hand-written vit_base()."""
    class _FakeHFConfig:
        hidden_size = 768
        num_attention_heads = 12
        num_hidden_layers = 12
        intermediate_size = 3072
        image_size = 224
        patch_size = 16
        num_labels = 1000

    cfg = ModelConfig.from_hf_config(_FakeHFConfig())
    assert cfg == ModelConfig.vit_base()


def test_from_hf_config_deit_tiny():
    class _FakeHFConfig:
        hidden_size = 192
        num_attention_heads = 3
        num_hidden_layers = 12
        intermediate_size = 768
        image_size = 224
        patch_size = 16
        num_labels = 1000

    cfg = ModelConfig.from_hf_config(_FakeHFConfig())
    assert cfg == ModelConfig.deit_tiny()
