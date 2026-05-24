#!/usr/bin/env python3
"""CLI: simulate program.bin with input image.

Supports two precision modes via ``--mode`` (must match the mode the
program was compiled with):

* ``w8a16`` (default, shipping) — reads FP16 patch embeddings,
  dispatches to ``SimulatorW8A16``, extracts FP16 logits from ABUF at
  the offset recorded in ``compiler_manifest['classifier_output']``.
* ``w8a32`` — reads FP32 patch embeddings (host-side Conv2d output),
  dispatches to ``SimulatorW8A32``, extracts FP32 logits from ABUF at
  the offset recorded in ``compiler_manifest['classifier_output']``.
"""
import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="TACCEL golden model simulator")
    parser.add_argument("program", help="Compiled program.bin")
    parser.add_argument("--input", help="Input image file (.npy or .bin)")
    parser.add_argument("--output", help="Output logits file (.npy)")
    parser.add_argument("--top-k", type=int, default=5, help="Show top-K predictions")
    parser.add_argument(
        "--mode",
        choices=["w8a16", "w8a32"],
        default="w8a16",
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
    else:
        from taccel.golden_model.simulator_w8a16 import SimulatorW8A16
        sim = SimulatorW8A16()
    sim.load_program(prog)
    state = sim.state

    if args.input:
        if args.mode == "w8a32":
            if args.input.endswith('.npy'):
                inp = np.load(args.input).astype(np.float32)
            else:
                inp = np.frombuffer(open(args.input, 'rb').read(), dtype=np.float32)
            dtype_label = "FP32"
        else:
            if args.input.endswith('.npy'):
                inp = np.load(args.input).astype(np.float16)
            else:
                inp = np.frombuffer(open(args.input, 'rb').read(), dtype=np.float16)
            dtype_label = "FP16"
        inp_bytes = inp.tobytes()
        state.dram[prog.input_offset:prog.input_offset + len(inp_bytes)] = inp_bytes
        print(f"Loaded {dtype_label} input: {inp.shape}")

    print("Running simulation...")
    count = sim.run(max_instructions=prog.insn_count + 10)
    print(f"  Executed {count} instructions, {state.cycle_count} cycles")

    co = prog.compiler_manifest["classifier_output"]
    if args.mode == "w8a32":
        logits_fp32 = np.frombuffer(
            state.abuf, dtype=np.float32,
            count=co["N_pad"], offset=co["offset_bytes"],
        )[:co["logical_cols"]].copy()
    else:
        logits_fp32 = np.frombuffer(
            state.abuf, dtype=np.float16,
            count=co["N_pad"], offset=co["offset_bytes"],
        )[:co["logical_cols"]].astype(np.float32)

    if args.output:
        np.save(args.output, logits_fp32)
        print(f"Saved logits to {args.output}")

    top_k = min(args.top_k, len(logits_fp32))
    top_indices = np.argsort(logits_fp32)[::-1][:top_k]
    print(f"\nTop-{top_k} predictions:")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. class {idx}: {logits_fp32[idx]:.2f}")


if __name__ == "__main__":
    main()
