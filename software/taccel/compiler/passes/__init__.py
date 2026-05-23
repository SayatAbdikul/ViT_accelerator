"""IR pass framework.

A *pass* is a function ``(graph, cfg, ctx) -> graph`` that consumes an
:class:`IRGraph`, possibly rewrites it, and returns the (same or new) graph.
The default pipeline is empty so DeiT-tiny compilation is byte-identical to
pre-pass behaviour; passes activate only when their analysis determines a
rewrite is needed (e.g. sequence tiling activates when the [seq, embed]
activation tensor would exceed ABUF capacity on its own — i.e. ViT-B/16
and larger).

The ``ctx`` dict is a free-form scratch space passes can use to communicate
with each other or with the caller. Currently it carries:

* ``ctx["calibration_scales"]`` — keyed by node name. Passes that synthesise
  new nodes (e.g. per-tile copies) must replicate the parent's scale entries
  so the codegen lookup ``calibration_scales[node.name]`` succeeds. The
  codegen falls back to stripping the ``_tileN_`` infix if the lookup misses,
  so this replication is best-effort, not mandatory.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Any

from ..ir import IRGraph
from ...model_config import ModelConfig

from .seq_tiling import (
    seq_tiling_pass,
    seq_tiling_pass_w8a32,
    seq_tiling_pass_w8a16,
)  # re-export


PassFn = Callable[[IRGraph, ModelConfig, Dict[str, Any]], IRGraph]


def default_pipeline() -> List[PassFn]:
    """Return the canonical pass pipeline.

    Currently only sequence tiling. Each pass is self-gated: if its analysis
    determines no rewrite is required (e.g. all activations fit in ABUF),
    it returns the graph unchanged.
    """
    return [seq_tiling_pass]


def default_pipeline_w8a32() -> List[PassFn]:
    """Pass pipeline for the W8A32 path.

    Uses the FP32-aware sequence-tiling policy (4× per-element bytes), which
    triggers tiling for both DeiT-tiny and ViT-B since a full FP32 residual
    exceeds the 128 KB ABUF in either case.
    """
    return [seq_tiling_pass_w8a32]


def default_pipeline_w8a16() -> List[PassFn]:
    """Pass pipeline for the W8A16 path.

    Uses the FP16-aware sequence-tiling policy (2× per-element bytes). The
    FP16 residual fits in ABUF on DeiT-tiny, but the FC1 per-tile cap still
    forces tiling — landing on the same tile_rows shape as the W8A32 path.
    """
    return [seq_tiling_pass_w8a16]


def run_passes(
    graph: IRGraph,
    cfg: ModelConfig,
    ctx: Optional[Dict[str, Any]] = None,
    pipeline: Optional[List[PassFn]] = None,
) -> IRGraph:
    """Apply the pass pipeline to ``graph``.

    Returns the (possibly rewritten) graph. ``ctx`` is mutated in place so
    callers can inspect what each pass did (e.g. replicated scales) and the
    next pass can build on prior pass state.
    """
    if ctx is None:
        ctx = {}
    if pipeline is None:
        pipeline = default_pipeline()
    for pass_fn in pipeline:
        graph = pass_fn(graph, cfg, ctx)
    return graph


__all__ = ["run_passes", "default_pipeline", "seq_tiling_pass", "PassFn"]
