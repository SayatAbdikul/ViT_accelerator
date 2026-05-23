#!/usr/bin/env python3
"""CLI: simulate program.bin with input image.

Supports two precision modes via ``--mode`` (must match the mode the
program was compiled with):

* ``w8a8`` (default) — reads INT8 patch embeddings, dispatches to
  ``Simulator``, extracts INT32 ACCUM logits and casts to FP32.
* ``w8a32`` — reads FP32 patch embeddings (host-side Conv2d output),
  dispatches to ``SimulatorW8A32``, extracts FP32 logits from ABUF at
  the offset recorded in ``compiler_manifest['classifier_output']``.
"""
import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taccel.compiler.graph_extract import EMBED_DIM, NUM_PATCHES
from taccel.compiler.tiler import pad_dim


def load_input_array(path: str) -> np.ndarray:
    """Load an INT8 input tensor from .npy or raw .bin."""
    if path.endswith('.npy'):
        arr = np.load(path)
    else:
        arr = np.frombuffer(open(path, 'rb').read(), dtype=np.int8)
    return np.asarray(arr, dtype=np.int8)


def _prepare_input_bytes(inp: np.ndarray) -> bytes:
    """Serialize an INT8 activation tensor, padding 2D rows to 16-byte width."""
    arr = np.asarray(inp, dtype=np.int8)
    if arr.ndim == 2:
        rows, cols = arr.shape
        cols_pad = pad_dim(cols)
        if cols_pad != cols:
            padded = np.zeros((rows, cols_pad), dtype=np.int8)
            padded[:, :cols] = arr
            arr = padded
        return arr.tobytes()
    return arr.reshape(-1).tobytes()


def write_runtime_inputs(state, program, inp: np.ndarray,
                         cls_input: np.ndarray | None = None,
                         folded_pos_embed: bool = False) -> None:
    """Place runtime inputs according to ProgramBinary metadata.

    Modern programs expect patch embeddings in DRAM at ``program.input_offset``.
    Legacy binaries leave that metadata at 0 and still consume ABUF directly.
    """
    input_bytes = _prepare_input_bytes(inp)
    input_offset = getattr(program, "input_offset", 0)
    if input_offset > 0:
        state.dram[input_offset:input_offset + len(input_bytes)] = input_bytes
    else:
        state.abuf[:len(input_bytes)] = input_bytes

    if cls_input is not None and getattr(program, "cls_token_dram_offset", 0) > 0:
        cls_arr = np.asarray(cls_input, dtype=np.int8)
        if cls_arr.ndim == 1:
            cls_arr = cls_arr.reshape(1, -1)
        cls_bytes = cls_arr[0, :EMBED_DIM].tobytes()
        cls_off = program.cls_token_dram_offset
        state.dram[cls_off:cls_off + len(cls_bytes)] = cls_bytes

    if not folded_pos_embed:
        return

    patch_pos_off = getattr(program, "pos_embed_patch_dram_offset", 0)
    if patch_pos_off > 0:
        arr = np.asarray(inp, dtype=np.int8)
        if arr.ndim == 2:
            patch_pos_size = arr.shape[0] * pad_dim(arr.shape[1])
        else:
            patch_pos_size = NUM_PATCHES * pad_dim(EMBED_DIM)
        state.dram[patch_pos_off:patch_pos_off + patch_pos_size] = bytes(patch_pos_size)

    if cls_input is not None and getattr(program, "pos_embed_cls_dram_offset", 0) > 0:
        cls_pos_off = program.pos_embed_cls_dram_offset
        state.dram[cls_pos_off:cls_pos_off + EMBED_DIM] = bytes(EMBED_DIM)


def main():
    parser = argparse.ArgumentParser(description="TACCEL golden model simulator")
    parser.add_argument("program", help="Compiled program.bin")
    parser.add_argument("--input", help="Input image file (.npy or .bin)")
    parser.add_argument(
        "--cls-input",
        help="Optional folded CLS row (.npy or .bin) written to cls_token_dram_offset when available",
    )
    parser.add_argument(
        "--folded-pos-embed",
        action="store_true",
        help="Zero folded position-embedding rows using ProgramBinary metadata",
    )
    parser.add_argument("--output", help="Output logits file (.npy)")
    parser.add_argument("--top-k", type=int, default=5, help="Show top-K predictions")
    parser.add_argument(
        "--mode",
        choices=["w8a8", "w8a32", "w8a16"],
        default="w8a8",
        help="Precision mode (must match the program's compile mode).",
    )
    args = parser.parse_args()

    print(f"Loading program {args.program}...")
    from taccel.assembler.assembler import ProgramBinary
    with open(args.program, 'rb') as f:
        raw = f.read()
    prog = ProgramBinary.from_bytes(raw)
    print(f"  {prog.insn_count} instructions, {len(prog.data):,} bytes data")

    if args.mode == "w8a32":
        from taccel.golden_model.simulator_w8a32 import SimulatorW8A32
        sim = SimulatorW8A32()
        sim.load_program(prog)
        state = sim.state
    elif args.mode == "w8a16":
        from taccel.golden_model.simulator_w8a16 import SimulatorW8A16
        sim = SimulatorW8A16()
        sim.load_program(prog)
        state = sim.state
    else:
        from taccel.golden_model import Simulator, MachineState
        state = MachineState(dram_data=prog.data)
        sim = Simulator(state)
        sim.load_program(prog)

    # Load input if provided
    if args.input:
        if args.mode == "w8a32":
            # FP32 patch embeddings (host-side Conv2d output).
            if args.input.endswith('.npy'):
                inp_fp32 = np.load(args.input).astype(np.float32)
            else:
                inp_fp32 = np.frombuffer(
                    open(args.input, 'rb').read(), dtype=np.float32
                )
            inp_bytes = inp_fp32.tobytes()
            state.dram[prog.input_offset:prog.input_offset + len(inp_bytes)] = inp_bytes
            print(f"Loaded FP32 input: {inp_fp32.shape}")
        elif args.mode == "w8a16":
            # FP16 patch embeddings (host-side Conv2d output, narrowed to FP16).
            if args.input.endswith('.npy'):
                inp_fp16 = np.load(args.input).astype(np.float16)
            else:
                inp_fp16 = np.frombuffer(
                    open(args.input, 'rb').read(), dtype=np.float16
                )
            inp_bytes = inp_fp16.tobytes()
            state.dram[prog.input_offset:prog.input_offset + len(inp_bytes)] = inp_bytes
            print(f"Loaded FP16 input: {inp_fp16.shape}")
        else:
            inp = load_input_array(args.input)
            cls_inp = load_input_array(args.cls_input) if args.cls_input else None
            write_runtime_inputs(
                state,
                prog,
                inp,
                cls_input=cls_inp,
                folded_pos_embed=args.folded_pos_embed,
            )
            print(f"Loaded input: {inp.shape}")
            if cls_inp is not None:
                print(f"Loaded CLS input: {cls_inp.shape}")

    print("Running simulation...")
    if args.mode in ("w8a32", "w8a16"):
        count = sim.run(max_instructions=prog.insn_count + 10)
    else:
        count = sim.run()
    print(f"  Executed {count} instructions, {state.cycle_count} cycles")

    if args.mode == "w8a32":
        co = prog.compiler_manifest["classifier_output"]
        logits_fp32 = np.frombuffer(
            state.abuf, dtype=np.float32,
            count=co["N_pad"], offset=co["offset_bytes"],
        )[:co["logical_cols"]].copy()
    elif args.mode == "w8a16":
        co = prog.compiler_manifest["classifier_output"]
        logits_fp32 = np.frombuffer(
            state.abuf, dtype=np.float16,
            count=co["N_pad"], offset=co["offset_bytes"],
        )[:co["logical_cols"]].astype(np.float32)
    else:
        # The compiler places classifier output at ACCUM[0] as INT32.
        logits_int32 = state.accum[:1000].copy()
        logits_fp32 = logits_int32.astype(np.float32)

    if args.output:
        np.save(args.output, logits_fp32)
        print(f"Saved logits to {args.output}")

    # Show top predictions
    top_k = min(args.top_k, 1000)
    top_indices = np.argsort(logits_fp32)[::-1][:top_k]
    print(f"\nTop-{top_k} predictions:")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. class {idx}: {logits_fp32[idx]:.2f}")


if __name__ == "__main__":
    main()
