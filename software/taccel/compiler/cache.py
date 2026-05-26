"""Disk cache for compiled :class:`ProgramBinary` artifacts.

A typical ``Compiler.compile_w8a16(state_dict)`` for DeiT-tiny takes
~5–7 s, dominated by per-weight quantize/dequant and codegen. The
compile is **fully deterministic** for a fixed
``(model_config, mode, state_dict)`` triple — we've verified this
directly: the smoke compiled DeiT-tiny twice and the resulting
``program.bin`` was byte-identical.

This module memoises that compile on disk. A cache key is derived from
the hashes of (a) the state_dict tensors (name + dtype + shape + bytes),
(b) the ModelConfig fields, and (c) the precision mode. On a hit we
restore via ``ProgramBinary.from_bytes`` and skip the compile entirely.

Cache location:
- ``$TACCEL_COMPILE_CACHE_DIR`` if set, else
- ``~/.cache/taccel/compile/`` (XDG-style).

Set ``TACCEL_NO_COMPILE_CACHE=1`` to bypass the cache (useful when
debugging compiler changes — otherwise stale entries would silently
serve old programs).
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

from ..assembler.assembler import ProgramBinary
from ..model_config import ModelConfig

if TYPE_CHECKING:
    pass


# Bump when the on-disk cache format or any compile-logic semantics change
# so old cache entries are invalidated automatically.
CACHE_FORMAT_VERSION = 1


def _default_cache_dir() -> Path:
    """Return the cache directory, respecting TACCEL_COMPILE_CACHE_DIR."""
    env = os.environ.get("TACCEL_COMPILE_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "taccel" / "compile"


def _hash_state_dict(state_dict: dict) -> str:
    """Stable hash of the state_dict's numpy-able tensors.

    Order-independent (sorts by tensor name) and skips entries that
    don't expose a numpy view — same predicate the compiler uses in
    its tensor walk."""
    h = hashlib.sha256()
    for name in sorted(state_dict.keys()):
        t = state_dict[name]
        if not hasattr(t, "numpy"):
            continue
        arr = t.numpy()
        # Cheap "name | dtype | shape" preamble lets us reject hash
        # collisions across different tensor layouts that happen to
        # produce identical byte sequences after reshape.
        h.update(name.encode("utf-8"))
        h.update(str(arr.dtype).encode("ascii"))
        h.update(repr(arr.shape).encode("ascii"))
        h.update(arr.tobytes())
    return h.hexdigest()


def _hash_config(cfg: ModelConfig, mode: str) -> str:
    """Stable hash of the (ModelConfig, mode) pair."""
    fields = tuple(sorted(dataclasses.asdict(cfg).items()))
    payload = f"v={CACHE_FORMAT_VERSION}|mode={mode}|cfg={fields}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_cache_key(cfg: ModelConfig, mode: str, state_dict: dict) -> str:
    """Return the on-disk cache key for this compile.

    Format: ``<config_hash[:16]>-<state_hash[:16]>``. Short enough to use
    as a filesystem path component, wide enough that collisions are
    astronomically improbable (96 bits of entropy total).
    """
    cfg_h = _hash_config(cfg, mode)
    sd_h = _hash_state_dict(state_dict)
    return f"{cfg_h[:16]}-{sd_h[:16]}"


def _disabled_via_env() -> bool:
    """Cache bypass for compiler-debugging sessions."""
    return os.environ.get("TACCEL_NO_COMPILE_CACHE", "").lower() in (
        "1", "true", "yes",
    )


def load_or_compile(
    cfg: ModelConfig,
    state_dict: dict,
    mode: Literal["w8a16", "w8a32"] = "w8a16",
    *,
    cache_dir: Optional[Path] = None,
    verbose: bool = False,
) -> ProgramBinary:
    """Return a ProgramBinary for this (cfg, mode, state_dict), using disk cache.

    On a cache hit, restores the previously compiled binary via
    :meth:`ProgramBinary.from_bytes` and returns it directly. On a miss
    (or when caching is disabled), runs the normal compile path and
    persists the result for future calls.
    """
    # Lazy import to avoid a hard cycle: Compiler imports from .codegen_*
    # which doesn't need this module, but a user of this module pulls in
    # the full compile graph.
    from .compiler import Compiler

    def _do_compile() -> ProgramBinary:
        compiler = Compiler(cfg=cfg, mode=mode)
        if mode == "w8a16":
            return compiler.compile_w8a16(state_dict)
        if mode == "w8a32":
            return compiler.compile_w8a32(state_dict)
        raise ValueError(f"unknown mode {mode!r}")

    if _disabled_via_env():
        if verbose:
            print("[compile_cache] disabled via TACCEL_NO_COMPILE_CACHE=1")
        return _do_compile()

    cache_dir = cache_dir or _default_cache_dir()
    key = compute_cache_key(cfg, mode, state_dict)
    cache_path = cache_dir / f"{key}.bin"

    if cache_path.exists():
        if verbose:
            print(f"[compile_cache] hit: {cache_path}")
        return ProgramBinary.from_bytes(cache_path.read_bytes())

    if verbose:
        print(f"[compile_cache] miss: {cache_path} — compiling")
    program = _do_compile()

    # Atomic write: stage into a sibling tmp file, then rename. Avoids a
    # half-written cache file leaking to other processes if we get
    # interrupted mid-write.
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".bin.tmp")
    tmp_path.write_bytes(program.to_bytes())
    tmp_path.rename(cache_path)
    if verbose:
        print(f"[compile_cache] wrote {cache_path} ({cache_path.stat().st_size:,} bytes)")
    return program
