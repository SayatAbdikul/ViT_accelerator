# Verilator build & test Makefile for TACCEL Phase 1
#
# Prerequisites:
#   verilator >= 5.0   (brew install verilator  OR  apt install verilator)
#   g++ / clang++ with C++17
#
# Usage:
#   make           — build + run all Phase 1 tests
#   make test_decode   — decode unit tests only
#   make test_control  — control unit / full-chip tests only
#   make clean     — remove generated objects

export LC_ALL := C
export LANG := C

VERILATOR     ?= verilator
CXX           ?= g++
CXXFLAGS      := -std=c++17 -O2 -Wall -Wno-unused-variable
# Parallel build: -j tells Verilator to fan out the C++ compile across all
# cores; --output-split caps generated translation-unit size so that fan-out
# actually scales. Override via `make NPROC=4 ...` on machines where you
# want to leave cores free.
NPROC         ?= $(shell nproc 2>/dev/null || echo 4)
# Parallel simulation: --threads partitions the generated eval across N
# threads at runtime. 4 is the default — leaves the other half of an 8-core
# / 16-thread box free for parallel runners or build steps. Set to 0 to
# disable multithreaded sim (single-threaded eval; smallest binary, useful
# for tiny tests). Override via `make SIM_THREADS=N ...`.
SIM_THREADS   ?= 4
ifneq ($(SIM_THREADS),0)
VTHREADS_FLAG := --threads $(SIM_THREADS)
else
VTHREADS_FLAG :=
endif
VFLAGS        := --cc --exe --build --timing \
                 -Wall -Wno-UNUSED -Wno-UNDRIVEN \
                 --x-assign fast --x-initial fast \
                 -j $(NPROC) --output-split 20000 \
                 $(VTHREADS_FLAG)
SYSTOLIC_ARCH_MODE ?= 1

SRC_DIR       := ../src
INCLUDE_DIR   := $(SRC_DIR)/include
SYSTOLIC_DIR  := $(SRC_DIR)/systolic
TB_DIR        := $(SRC_DIR)/tb
BUILD_DIR     := build

# Source file lists
PKG           := $(INCLUDE_DIR)/taccel_pkg.sv
FP32_PKG      := $(INCLUDE_DIR)/fp32_prim_pkg.sv
DECODE_SV     := $(PKG) $(SRC_DIR)/decode_unit.sv $(TB_DIR)/tb_decode_unit.sv
CONTROL_SV    := $(PKG) \
                 $(FP32_PKG) \
                 $(SRC_DIR)/decode_unit.sv \
                 $(SRC_DIR)/fetch_unit.sv \
                 $(SRC_DIR)/control_unit.sv \
                 $(SRC_DIR)/blocking_helper_engine.sv \
                 $(SRC_DIR)/sfu_engine.sv \
                 $(SRC_DIR)/memory/register_file.sv \
                 $(SRC_DIR)/memory/sram_dp.sv \
                 $(SRC_DIR)/memory/sram_subsystem.sv \
				 $(SRC_DIR)/systolic/systolic_pe.sv \
				 $(SRC_DIR)/systolic/systolic_array.sv \
				 $(SRC_DIR)/systolic/systolic_controller.sv \
                 $(SRC_DIR)/dma_engine.sv \
                 $(SRC_DIR)/taccel_top.sv

.PHONY: all run_program test_decode test_control test_dma test_helpers test_fp32_prims test_sfu \
        test_systolic test_systolic_array_chained test_systolic_chained \
        test_systolic_query test_systolic_qkt synth_gate lint clean

all: test_decode test_control test_dma test_helpers test_fp32_prims test_sfu test_systolic

# -----------------------------------------------------------------------
# synth_gate: fail if non-synthesizable constructs (DPI-C, SystemVerilog
# `real`, $realtobits/$bitstoreal/$itor/$rtoi) appear in the synthesizable
# RTL set (CONTROL_SV — packages + engines + memory + systolic + dma + top;
# the tb/ harness is intentionally excluded). Line comments are stripped
# before matching. This gate FAILS today by design and is the burndown
# target for the DPI/real removal work (see include/ARITH_CONTRACT.md).
# -----------------------------------------------------------------------
SYNTH_GATE_SRC := $(sort $(CONTROL_SV))
SYNTH_GATE_RE  := import[[:space:]]+"DPI-C"|(^|[^A-Za-z0-9_])real([^A-Za-z0-9_]|$$)|\$$realtobits|\$$bitstoreal|\$$itor|\$$rtoi|\$$shortrealtobits

synth_gate:
	@echo "--- SYNTH_BUILD gate: scanning synthesizable RTL ---"
	@fail=0; for f in $(SYNTH_GATE_SRC); do \
	  hits=$$(sed 's://.*::' $$f | grep -nE '$(SYNTH_GATE_RE)' || true); \
	  if [ -n "$$hits" ]; then echo "[SYNTH-GATE] $$f"; echo "$$hits"; fail=1; fi; \
	done; \
	if [ $$fail -ne 0 ]; then echo "SYNTH_BUILD gate FAILED (non-synthesizable constructs above)"; exit 1; fi; \
	echo "SYNTH_BUILD gate PASSED"

lint: synth_gate
	$(VERILATOR) --lint-only -Wall -Wno-UNUSED -Wno-UNDRIVEN \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) --top-module taccel_top $(CONTROL_SV)
	@echo "--- lint OK ---"

# -----------------------------------------------------------------------
# run_program: build the full-program Verilator runner
# -----------------------------------------------------------------------
$(BUILD_DIR)/run_program/Vtaccel_top: $(CONTROL_SV) run_program.cpp
	mkdir -p $(BUILD_DIR)/run_program
	$(VERILATOR) $(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=$(SYSTOLIC_ARCH_MODE) \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module taccel_top \
	  --Mdir $(BUILD_DIR)/run_program \
	  $(CONTROL_SV) \
	  run_program.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/run_program -f Vtaccel_top.mk

run_program: $(BUILD_DIR)/run_program/Vtaccel_top

# -----------------------------------------------------------------------
# test_decode: verilate tb_decode_unit + compile + run
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_decode/Vtb_decode_unit: $(DECODE_SV) test_decode.cpp
	mkdir -p $(BUILD_DIR)/test_decode
	$(VERILATOR) $(VFLAGS) \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module tb_decode_unit \
	  --Mdir $(BUILD_DIR)/test_decode \
	  $(DECODE_SV) \
	  test_decode.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_decode -f Vtb_decode_unit.mk

test_decode: $(BUILD_DIR)/test_decode/Vtb_decode_unit
	@echo "--- Running decode unit tests ---"
	$(BUILD_DIR)/test_decode/Vtb_decode_unit

# -----------------------------------------------------------------------
# test_control: verilate taccel_top + compile + run
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_control/Vtaccel_top: $(CONTROL_SV) test_control.cpp
	mkdir -p $(BUILD_DIR)/test_control
	$(VERILATOR) $(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=$(SYSTOLIC_ARCH_MODE) \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module taccel_top \
	  --Mdir $(BUILD_DIR)/test_control \
	  $(CONTROL_SV) \
	  test_control.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_control -f Vtaccel_top.mk

test_control: $(BUILD_DIR)/test_control/Vtaccel_top
	@echo "--- Running control unit tests ---"
	$(BUILD_DIR)/test_control/Vtaccel_top

# -----------------------------------------------------------------------
# test_dma: verilate taccel_top (with DMA engine) + compile + run
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_dma/Vtaccel_top: $(CONTROL_SV) test_dma.cpp
	mkdir -p $(BUILD_DIR)/test_dma
	$(VERILATOR) $(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=$(SYSTOLIC_ARCH_MODE) \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module taccel_top \
	  --Mdir $(BUILD_DIR)/test_dma \
	  $(CONTROL_SV) \
	  test_dma.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_dma -f Vtaccel_top.mk

test_dma: $(BUILD_DIR)/test_dma/Vtaccel_top
	@echo "--- Running DMA tests ---"
	$(BUILD_DIR)/test_dma/Vtaccel_top

# -----------------------------------------------------------------------
# test_helpers: verilate taccel_top (with helper engine) + compile + run
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_helpers/Vtaccel_top: $(CONTROL_SV) test_helpers.cpp
	mkdir -p $(BUILD_DIR)/test_helpers
	$(VERILATOR) $(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=$(SYSTOLIC_ARCH_MODE) \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module taccel_top \
	  --Mdir $(BUILD_DIR)/test_helpers \
	  $(CONTROL_SV) \
	  test_helpers.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_helpers -f Vtaccel_top.mk

test_helpers: $(BUILD_DIR)/test_helpers/Vtaccel_top
	@echo "--- Running helper engine tests ---"
	$(BUILD_DIR)/test_helpers/Vtaccel_top

# -----------------------------------------------------------------------
# test_fp32_prims: verilate standalone FP32 primitive package exerciser
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_fp32_prims/Vtb_fp32_prim: $(FP32_PKG) $(TB_DIR)/tb_fp32_prim.sv test_fp32_prims.cpp
	mkdir -p $(BUILD_DIR)/test_fp32_prims
	$(VERILATOR) $(VFLAGS) \
	  -I$(INCLUDE_DIR) \
	  --top-module tb_fp32_prim \
	  --Mdir $(BUILD_DIR)/test_fp32_prims \
	  $(FP32_PKG) \
	  $(TB_DIR)/tb_fp32_prim.sv \
	  test_fp32_prims.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_fp32_prims -f Vtb_fp32_prim.mk

test_fp32_prims: $(BUILD_DIR)/test_fp32_prims/Vtb_fp32_prim
	@echo "--- Running FP32 primitive tests ---"
	$(BUILD_DIR)/test_fp32_prims/Vtb_fp32_prim

# -----------------------------------------------------------------------
# test_sfu: verilate taccel_top (with Stage D SFU) + compile + run
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_sfu/Vtaccel_top: $(CONTROL_SV) test_sfu.cpp
	mkdir -p $(BUILD_DIR)/test_sfu
	$(VERILATOR) $(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=$(SYSTOLIC_ARCH_MODE) \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module taccel_top \
	  --Mdir $(BUILD_DIR)/test_sfu \
	  $(CONTROL_SV) \
	  test_sfu.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_sfu -f Vtaccel_top.mk

test_sfu: $(BUILD_DIR)/test_sfu/Vtaccel_top
	@echo "--- Running SFU tests ---"
	$(BUILD_DIR)/test_sfu/Vtaccel_top

# -----------------------------------------------------------------------
# test_systolic: verilate taccel_top (with systolic) + compile + run
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_systolic/Vtaccel_top: $(CONTROL_SV) test_systolic.cpp
	mkdir -p $(BUILD_DIR)/test_systolic
	$(VERILATOR) $(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=$(SYSTOLIC_ARCH_MODE) \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module taccel_top \
	  --Mdir $(BUILD_DIR)/test_systolic \
	  $(CONTROL_SV) \
	  test_systolic.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_systolic -f Vtaccel_top.mk

test_systolic: $(BUILD_DIR)/test_systolic/Vtaccel_top
	@echo "--- Running systolic tests ---"
	$(BUILD_DIR)/test_systolic/Vtaccel_top

# -----------------------------------------------------------------------
# test_systolic_array_chained: unit-test the chained mesh directly
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_systolic_array_chained/Vsystolic_array: \
		$(PKG) $(SRC_DIR)/systolic/systolic_pe.sv $(SRC_DIR)/systolic/systolic_array.sv \
		test_systolic_array_chained.cpp
	mkdir -p $(BUILD_DIR)/test_systolic_array_chained
	$(VERILATOR) $(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=1 \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module systolic_array \
	  --Mdir $(BUILD_DIR)/test_systolic_array_chained \
	  $(PKG) \
	  $(SRC_DIR)/systolic/systolic_pe.sv \
	  $(SRC_DIR)/systolic/systolic_array.sv \
	  test_systolic_array_chained.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_systolic_array_chained -f Vsystolic_array.mk

test_systolic_array_chained: $(BUILD_DIR)/test_systolic_array_chained/Vsystolic_array
	@echo "--- Running chained systolic array tests ---"
	$(BUILD_DIR)/test_systolic_array_chained/Vsystolic_array

# -----------------------------------------------------------------------
# test_systolic_chained: top-level chained MATMUL verification
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_systolic_chained/Vtaccel_top: $(CONTROL_SV) test_systolic_chained.cpp
	mkdir -p $(BUILD_DIR)/test_systolic_chained
	$(VERILATOR) $(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=1 \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module taccel_top \
	  --Mdir $(BUILD_DIR)/test_systolic_chained \
	  $(CONTROL_SV) \
	  test_systolic_chained.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_systolic_chained -f Vtaccel_top.mk

test_systolic_chained: $(BUILD_DIR)/test_systolic_chained/Vtaccel_top
	@echo "--- Running chained top-level systolic tests ---"
	$(BUILD_DIR)/test_systolic_chained/Vtaccel_top

# -----------------------------------------------------------------------
# test_systolic_query: focused chained query-projection regression
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_systolic_query/Vtaccel_top: $(CONTROL_SV) test_systolic_query.cpp
	mkdir -p $(BUILD_DIR)/test_systolic_query
	$(VERILATOR) $(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=1 \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module taccel_top \
	  --Mdir $(BUILD_DIR)/test_systolic_query \
	  $(CONTROL_SV) \
	  test_systolic_query.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_systolic_query -f Vtaccel_top.mk

test_systolic_query: $(BUILD_DIR)/test_systolic_query/Vtaccel_top
	@echo "--- Running focused chained query systolic regression ---"
	$(BUILD_DIR)/test_systolic_query/Vtaccel_top

# -----------------------------------------------------------------------
# test_systolic_qkt: focused QK^T prep + matmul regressions
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_systolic_qkt/Vtaccel_top: $(CONTROL_SV) test_systolic_qkt.cpp
	mkdir -p $(BUILD_DIR)/test_systolic_qkt
	$(VERILATOR) $(VFLAGS) \
	  -GSYSTOLIC_ARCH_MODE=1 \
	  -I$(INCLUDE_DIR) -I$(SYSTOLIC_DIR) \
	  --top-module taccel_top \
	  --Mdir $(BUILD_DIR)/test_systolic_qkt \
	  $(CONTROL_SV) \
	  test_systolic_qkt.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_systolic_qkt -f Vtaccel_top.mk

test_systolic_qkt: $(BUILD_DIR)/test_systolic_qkt/Vtaccel_top
	@echo "--- Running focused QK^T regressions ---"
	$(BUILD_DIR)/test_systolic_qkt/Vtaccel_top

# -----------------------------------------------------------------------
clean:
	rm -rf $(BUILD_DIR)
