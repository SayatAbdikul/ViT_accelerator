#!/usr/bin/env python3
"""CLI: pytorch_model.bin → program.bin.

Supports two precision modes via ``--mode``:

* ``w8a16`` (default) — INT8 weights (per-channel) dequantized into FP16
  DRAM, FP16 activations, FP32 accumulators. Shipping mode.
* ``w8a32`` — INT8 weights (per-channel) dequantized into FP32 DRAM, FP32
  activations, FP32 accumulators. Software-only weight-quant ceiling
  reference; doubles the dequant-weight DRAM footprint vs w8a16.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_NAME = "facebook/deit-tiny-patch16-224"


def main():
    parser = argparse.ArgumentParser(description="TACCEL compiler: PyTorch model → ProgramBinary")
    parser.add_argument("--weights", required=True, help="PyTorch weights file (pytorch_model.bin)")
    parser.add_argument("--model", default="deit-tiny",
                        choices=["deit-tiny"], help="Model architecture")
    parser.add_argument("-o", "--output", default="program.bin", help="Output .bin file")
    parser.add_argument(
        "--mode",
        choices=["w8a16", "w8a32"],
        default="w8a16",
        help="Precision mode (see module docstring). w8a16 is the default "
             "shipping path; w8a32 doubles dequant DRAM but is the FP32 "
             "weight-quant ceiling reference.",
    )
    args = parser.parse_args()

    print(f"Loading weights from {args.weights}...")
    import torch
    state_dict = torch.load(args.weights, map_location="cpu", weights_only=False)
    if hasattr(state_dict, 'items') is False:
        state_dict = state_dict.state_dict()

    from taccel.compiler import Compiler
    from taccel.model_config import ModelConfig

    if args.mode == "w8a32":
        print(f"Compiling {args.model} in W8A32 mode...")
        compiler = Compiler(cfg=ModelConfig.deit_tiny(), mode="w8a32")
        prog = compiler.compile_w8a32(state_dict)
    else:
        print(f"Compiling {args.model} in W8A16 mode...")
        compiler = Compiler(cfg=ModelConfig.deit_tiny(), mode="w8a16")
        prog = compiler.compile_w8a16(state_dict)

    with open(args.output, 'wb') as f:
        f.write(prog.to_bytes())

    print(f"\nCompilation complete:")
    print(f"  Mode: {args.mode}")
    print(f"  Instructions: {prog.insn_count}")
    print(f"  Instruction bytes: {len(prog.instructions):,}")
    print(f"  Data (weights): {len(prog.data):,} bytes")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
