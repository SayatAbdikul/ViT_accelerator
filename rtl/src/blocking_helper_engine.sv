// Blocking helper engine for the W8A16 datapath.
//
// Supported operations (W8A16 contract):
//   - BUF_COPY  : flat byte copy / FP16-element-aware transpose
//   - VADD      : FP16 + FP16 -> FP16 (ABUF) or FP32 + FP16-broadcast -> FP32
//                 (ACCUM bias / attention-mask add)
//   - SCALE_MUL : FP16 widen / FP32 in -> FP32 mul -> FP16 narrow (ABUF dst)
//                 or stay FP32 (ACCUM dst)
//
// Removed in W8A16 (was W8A8):
//   - REQUANT (0x0B), REQUANT_PC (0x11), DEQUANT_ADD (0x13). These remain
//     legal ISA encodings but are unsupported here; dispatching them raises
//     FAULT_UNSUPPORTED_OP. Phase 5 hoists the rejection into decode_unit;
//     until then this engine catches the dispatch.
//
// Datapath:
//   - ABUF rows hold 8 FP16 elements per 128-bit row (was 16 INT8).
//   - ACCUM rows hold 4 FP32 elements per 128-bit row (same 4x32-bit slots
//     as INT32; bit interpretation flipped to FP32).
//   - Geometry: fp16 row count = m_rows * n_tiles * 2  (= n_fp16_chunks),
//     fp32 row count = m_rows * n_tiles * 4  (= n_chunks_i32, unchanged).
//
// The helper engine is architecturally blocking. Control dispatches it
// through helper_dispatch and waits in S_DISP_WAIT until helper_busy drops.
//
// It owns both SRAM ports while active:
//   - port A for reads and writes
//   - port B for a second read stream when an operation needs two sources

`ifndef BLOCKING_HELPER_ENGINE_SV
`define BLOCKING_HELPER_ENGINE_SV

`include "taccel_pkg.sv"

module blocking_helper_engine
  import taccel_pkg::*;
  import fp32_prim_pkg::*;
(
  input  logic         clk,
  input  logic         rst_n,

  input  logic         dispatch,
  input  logic [4:0]   opcode,
  input  logic [1:0]   src1_buf,
  input  logic [15:0]  src1_off,
  input  logic [1:0]   src2_buf,
  input  logic [15:0]  src2_off,
  input  logic [1:0]   dst_buf,
  input  logic [15:0]  dst_off,
  input  logic [3:0]   sreg,
  input  logic [15:0]  b_length,
  input  logic [5:0]   b_src_rows,
  input  logic         b_transpose,
  input  logic [9:0]   tile_m,
  input  logic [9:0]   tile_n,
  input  logic [15:0]  scale0_data,
  /* verilator lint_off UNUSED */
  input  logic [15:0]  scale1_data,
  /* verilator lint_on UNUSED */

  output logic         helper_busy,
  output logic         helper_fault,
  output logic [3:0]   helper_fault_code,

  output logic         sram_a_en,
  output logic         sram_a_we,
  output logic [1:0]   sram_a_buf,
  output logic [15:0]  sram_a_row,
  output logic [127:0] sram_a_wdata,
  input  logic [127:0] sram_a_rdata,
  input  logic         sram_a_fault,

  output logic         sram_b_en,
  output logic [1:0]   sram_b_buf,
  output logic [15:0]  sram_b_row,
  input  logic [127:0] sram_b_rdata,
  input  logic         sram_b_fault
);

  // State groups:
  //   H_FLAT_*  : BUF_COPY flat copy / memmove (byte-oriented)
  //   H_TSRC_*  : BUF_COPY transpose source gather (FP16-element tile)
  //   H_TDST_*  : BUF_COPY transpose destination scatter
  //   H_V16_*   : FP16 ABUF VADD (FP16 + FP16 -> FP16)
  //   H_VB_*    : ACCUM VADD bias-row load (one FP16 row, reused across M)
  //   H_VACC_*  : ACCUM VADD FP32 + FP16-widened-broadcast -> FP32
  //   H_SM1_*   : SCALE_MUL one-read-one-write (ABUF->ABUF or ACCUM->ACCUM)
  //   H_SM2_*   : SCALE_MUL two-reads-one-write (ACCUM->ABUF narrowing)
  typedef enum logic [4:0] {
    H_IDLE        = 5'd0,
    H_FLAT_READ   = 5'd1,
    H_FLAT_WRITE  = 5'd2,
    H_TSRC_REQ    = 5'd3,
    H_TSRC_LATCH  = 5'd4,
    H_TDST_WRITE  = 5'd5,
    H_V16_READ    = 5'd6,
    H_V16_WRITE   = 5'd7,
    H_VB_REQ      = 5'd8,
    H_VB_LATCH    = 5'd9,
    H_VACC_READ   = 5'd10,
    H_VACC_WRITE  = 5'd11,
    H_SM1_REQ     = 5'd12,
    H_SM1_WRITE   = 5'd13,
    H_SM2_REQ     = 5'd14,
    H_SM2_LATCH   = 5'd15,
    H_SM2_WRITE   = 5'd16,
    H_FAULT       = 5'd17
  } helper_state_t;

  helper_state_t state;

  // Latched instruction parameters.
  logic [4:0]   opcode_q;
  logic [1:0]   src1_buf_q, src2_buf_q, dst_buf_q;
  logic [15:0]  src1_off_q, src2_off_q, dst_off_q;
  /* verilator lint_off UNUSED */
  logic [3:0]   sreg_q;
  /* verilator lint_on UNUSED */
  logic [15:0]  b_length_q;
  logic [5:0]   b_src_rows_q;
  /* verilator lint_off UNUSED */
  logic         b_transpose_q;
  /* verilator lint_on UNUSED */
  logic [14:0]  m_rows_q;
  logic [10:0]  n_tiles_q;
  logic [11:0]  n_fp16_chunks_q;
  logic [12:0]  n_chunks_i32_q;
  logic [3:0]   fault_code_r;

  // FP16 scale latched as FP32 once at dispatch — SCALE_MUL stays FP32-internal.
  fp32_t        scale_fp32_q;

  // Flat-copy / V16 step counter.
  logic [31:0]  step_idx_q;
  logic         flat_backward_q;

  // Transpose state. Inner tile is 8x8 FP16 elements (one 128-bit SRAM row
  // per tile row), so dimensions index FP16 elements, not bytes.
  logic [15:0]  trans_row_count_q;
  logic [15:0]  trans_cols_q;
  logic [15:0]  trans_rbase_q;
  logic [15:0]  trans_cbase_q;
  logic [3:0]   trans_height_q;
  logic [3:0]   trans_width_q;
  logic [3:0]   trans_src_row_idx_q;
  logic [3:0]   trans_dst_row_idx_q;
  logic [127:0] trans_scratch_q [0:7];

  // ACCUM bias / VACC scheduling state.
  // bias_chunk_q counts in FP16-row units (each = 8 columns).
  // c4_half_q selects which 4-column half of the FP16 bias row maps to the
  // current ACCUM 4-column chunk: c4 = 2*bias_chunk_q + c4_half_q.
  logic [11:0]  bias_chunk_q;
  logic         c4_half_q;
  logic [14:0]  bias_row_idx_q;
  logic [127:0] bias_data_q;

  // SCALE_MUL ACCUM->ABUF two-row latch (mirror of SFU GELU FP32->FP16).
  // Both rows must be registered because sram_b_rdata is only valid in the
  // cycle immediately following its read request.
  logic         sm_part_q;
  logic [127:0] sm_row0_q;
  logic [127:0] sm_row1_q;

  // Element accessors.
  function automatic logic [7:0] get_byte(
    input logic [127:0] row,
    input integer       idx
  );
    begin
      get_byte = row[(idx * 8) +: 8];
    end
  endfunction

  function automatic logic [15:0] get_u16(
    input logic [127:0] row,
    input integer       idx
  );
    begin
      get_u16 = row[(idx * 16) +: 16];
    end
  endfunction

  function automatic logic [31:0] get_u32(
    input logic [127:0] row,
    input integer       idx
  );
    begin
      get_u32 = row[(idx * 32) +: 32];
    end
  endfunction

  // Dispatch-time geometry.
  // n_fp16_chunks_w = n_tiles * 2 (each 16-col tile = 2 FP16 rows of 8 cols).
  // n_chunks_i32_w  = n_tiles * 4 (each 16-col tile = 4 FP32 rows of 4 cols).
  logic [14:0] dispatch_m_rows_w;
  logic [10:0] dispatch_n_tiles_w;
  logic [11:0] dispatch_n_fp16_chunks_w;
  logic [12:0] dispatch_n_chunks_i32_w;
  logic [31:0] dispatch_fp16_units_w;
  logic [31:0] dispatch_fp32_units_w;
  logic [31:0] dispatch_copy_units_w;
  logic [15:0] dispatch_src_rows_w;
  logic [15:0] dispatch_trans_byte_cols_w;
  logic [15:0] dispatch_trans_elem_cols_w;
  logic [15:0] dispatch_src_buf_rows_w;
  logic [15:0] dispatch_src2_buf_rows_w;
  logic [15:0] dispatch_dst_buf_rows_w;
  logic        dispatch_unsupported_w;
  logic        dispatch_sram_oob_w;
  logic        dispatch_is_vadd_fp16_w;
  logic        dispatch_is_vadd_bias_w;
  logic        dispatch_is_sm_fp16_w;     // ABUF -> ABUF
  logic        dispatch_is_sm_fp32_w;     // ACCUM -> ACCUM
  logic        dispatch_is_sm_acc2fp16_w; // ACCUM -> ABUF (narrow)
  logic        dispatch_same_buf_overlap_w;
  logic        dispatch_trans_misaligned_w;

  assign dispatch_m_rows_w        = ({5'h0, tile_m} + 15'd1) << 4;
  assign dispatch_n_tiles_w       = {1'b0, tile_n} + 11'd1;
  assign dispatch_n_fp16_chunks_w = {1'b0, dispatch_n_tiles_w} << 1;
  assign dispatch_n_chunks_i32_w  = {2'b0, dispatch_n_tiles_w} << 2;
  assign dispatch_fp16_units_w    = {17'h0, dispatch_n_fp16_chunks_w} * {17'h0, dispatch_m_rows_w};
  assign dispatch_fp32_units_w    = {19'h0, dispatch_n_chunks_i32_w} * {17'h0, dispatch_m_rows_w};
  assign dispatch_copy_units_w    = {16'h0, b_length};
  assign dispatch_src_rows_w      = {6'h0, b_src_rows, 4'h0};
  assign dispatch_trans_byte_cols_w =
      (b_src_rows == 6'h0) ? 16'h0 : (b_length / {10'h0, b_src_rows});
  assign dispatch_trans_elem_cols_w = dispatch_trans_byte_cols_w >> 1;
  assign dispatch_src_buf_rows_w  = buf_rows(src1_buf);
  assign dispatch_src2_buf_rows_w = buf_rows(src2_buf);
  assign dispatch_dst_buf_rows_w  = buf_rows(dst_buf);

  assign dispatch_is_vadd_fp16_w =
      (src1_buf == BUF_ABUF) &&
      ((src2_buf == BUF_ABUF) || (src2_buf == BUF_WBUF)) &&
      (dst_buf == BUF_ABUF);
  assign dispatch_is_vadd_bias_w =
      (src1_buf == BUF_ACCUM) &&
      (src2_buf == BUF_WBUF) &&
      (dst_buf == BUF_ACCUM);

  // FP16 modes accept any ABUF/WBUF combination on src1 and dst (matches the
  // permissive _read_act_fp32 / _write_act paths in simulator_w8a16).
  assign dispatch_is_sm_fp16_w =
      (src1_buf != BUF_ACCUM) && (dst_buf != BUF_ACCUM);
  assign dispatch_is_sm_fp32_w =
      (src1_buf == BUF_ACCUM) && (dst_buf == BUF_ACCUM);
  assign dispatch_is_sm_acc2fp16_w =
      (src1_buf == BUF_ACCUM) && (dst_buf != BUF_ACCUM);

  assign dispatch_same_buf_overlap_w =
      (src1_buf == dst_buf) &&
      ({16'h0, src1_off} < ({16'h0, dst_off} + dispatch_copy_units_w)) &&
      ({16'h0, dst_off} < ({16'h0, src1_off} + dispatch_copy_units_w));

  // Transpose alignment: byte_cols must be even (FP16 = 2 bytes), and both
  // element dimensions must be multiples of 8 so the inner tile is full 8x8.
  // The W8A16 codegen always emits 16-aligned dims, so this is non-restrictive
  // for the shipping toolchain.
  assign dispatch_trans_misaligned_w =
      (dispatch_trans_byte_cols_w[0] != 1'b0) ||
      (dispatch_trans_elem_cols_w[2:0] != 3'h0) ||
      (dispatch_src_rows_w[2:0] != 3'h0);

  // Dispatch-time validation. Reject unsupported mode combinations and whole
  // SRAM ranges before touching memory.
  always_comb begin
    dispatch_unsupported_w = 1'b0;
    dispatch_sram_oob_w    = 1'b0;

    case (opcode)
      OP_BUF_COPY: begin
        dispatch_sram_oob_w =
            ({1'b0, src1_off} + {1'b0, b_length} > {1'b0, dispatch_src_buf_rows_w}) ||
            ({1'b0, dst_off}  + {1'b0, b_length} > {1'b0, dispatch_dst_buf_rows_w});

        if (b_transpose) begin
          if (b_length == 16'h0)
            dispatch_unsupported_w = 1'b0;
          else if ((b_src_rows == 6'h0) ||
                   ((b_length % {10'h0, b_src_rows}) != 16'h0) ||
                   (src1_buf == dst_buf) ||
                   dispatch_trans_misaligned_w)
            dispatch_unsupported_w = 1'b1;
        end
      end

      OP_VADD: begin
        if (dispatch_is_vadd_fp16_w) begin
          dispatch_sram_oob_w =
              ({16'h0, src1_off} + dispatch_fp16_units_w > {16'h0, dispatch_src_buf_rows_w}) ||
              ({16'h0, src2_off} + dispatch_fp16_units_w > {16'h0, dispatch_src2_buf_rows_w}) ||
              ({16'h0, dst_off}  + dispatch_fp16_units_w > {16'h0, dispatch_dst_buf_rows_w});
        end else if (dispatch_is_vadd_bias_w) begin
          dispatch_sram_oob_w =
              ({16'h0, src1_off} + dispatch_fp32_units_w > {16'h0, dispatch_src_buf_rows_w}) ||
              ({16'h0, src2_off} + {20'h0, dispatch_n_fp16_chunks_w} > {16'h0, dispatch_src2_buf_rows_w}) ||
              ({16'h0, dst_off}  + dispatch_fp32_units_w > {16'h0, dispatch_dst_buf_rows_w});
        end else begin
          dispatch_unsupported_w = 1'b1;
        end
      end

      OP_SCALE_MUL: begin
        if (dispatch_is_sm_fp16_w) begin
          dispatch_sram_oob_w =
              ({16'h0, src1_off} + dispatch_fp16_units_w > {16'h0, dispatch_src_buf_rows_w}) ||
              ({16'h0, dst_off}  + dispatch_fp16_units_w > {16'h0, dispatch_dst_buf_rows_w});
        end else if (dispatch_is_sm_fp32_w) begin
          dispatch_sram_oob_w =
              ({16'h0, src1_off} + dispatch_fp32_units_w > {16'h0, dispatch_src_buf_rows_w}) ||
              ({16'h0, dst_off}  + dispatch_fp32_units_w > {16'h0, dispatch_dst_buf_rows_w});
        end else if (dispatch_is_sm_acc2fp16_w) begin
          dispatch_sram_oob_w =
              ({16'h0, src1_off} + dispatch_fp32_units_w > {16'h0, dispatch_src_buf_rows_w}) ||
              ({16'h0, dst_off}  + dispatch_fp16_units_w > {16'h0, dispatch_dst_buf_rows_w});
        end else begin
          dispatch_unsupported_w = 1'b1;
        end
      end

      // REQUANT / REQUANT_PC / DEQUANT_ADD: unsupported in W8A16.
      // Falling through default raises FAULT_UNSUPPORTED_OP.
      default:
        dispatch_unsupported_w = 1'b1;
    endcase
  end

  // Flat-copy addressing.
  logic [15:0] flat_src_row_w;
  logic [15:0] flat_dst_row_w;
  assign flat_src_row_w =
      flat_backward_q ? (src1_off_q + b_length_q - 16'(step_idx_q) - 16'h1)
                      : (src1_off_q + 16'(step_idx_q));
  assign flat_dst_row_w =
      flat_backward_q ? (dst_off_q + b_length_q - 16'(step_idx_q) - 16'h1)
                      : (dst_off_q + 16'(step_idx_q));

  // Transpose addressing (FP16-element granularity).
  // Each source FP16 row spans (elem_cols / 8) SRAM rows. A cbase increment
  // of 8 elements advances one SRAM row. Each block reads one 128-bit row
  // per source row (8 FP16 = one full row, no spanning).
  logic [15:0] trans_src_row_w;
  logic [15:0] trans_dst_row_w;
  logic [15:0] trans_elem_cols_q;
  assign trans_src_row_w =
      src1_off_q +
      ((trans_rbase_q + {12'h0, trans_src_row_idx_q}) * (trans_elem_cols_q >> 3)) +
      (trans_cbase_q >> 3);
  assign trans_dst_row_w =
      dst_off_q +
      ((trans_cbase_q + {12'h0, trans_dst_row_idx_q}) * (trans_row_count_q >> 3)) +
      (trans_rbase_q >> 3);

  // Transpose destination row data: gather column from 8 scratch entries.
  logic [127:0] trans_dst_data_w;
  always_comb begin
    trans_dst_data_w = 128'h0;
    for (int j = 0; j < 8; j++) begin
      trans_dst_data_w[(j * 16) +: 16] =
          trans_scratch_q[j][(integer'(trans_dst_row_idx_q) * 16) +: 16];
    end
  end

  // V16 (FP16 ABUF VADD) addressing.
  logic [15:0] v16_src1_row_w;
  logic [15:0] v16_src2_row_w;
  logic [15:0] v16_dst_row_w;
  assign v16_src1_row_w = src1_off_q + 16'(step_idx_q);
  assign v16_src2_row_w = src2_off_q + 16'(step_idx_q);
  assign v16_dst_row_w  = dst_off_q + 16'(step_idx_q);

  // V16 (FP16 ABUF VADD) compute: 8 lanes per row, widen + add + narrow.
  // sram_a_rdata holds src2 (latched from H_V16_READ port A request),
  // sram_b_rdata holds src1.
  logic [127:0] v16_write_data_w;
  always_comb begin
    v16_write_data_w = 128'h0;
    for (int lane = 0; lane < 8; lane++)
      v16_write_data_w[(lane * 16) +: 16] = fp32_to_fp16_bits(
          fp32_add_bits(fp32_from_fp16_bits(get_u16(sram_b_rdata, lane)),
                        fp32_from_fp16_bits(get_u16(sram_a_rdata, lane))));
  end

  // ACCUM bias VADD addressing.
  // c4 = 2 * bias_chunk_q + c4_half_q (column-of-4 index inside the M row).
  // bias FP16 row index = bias_chunk_q (each FP16 row spans 8 columns,
  // covering 2 ACCUM 4-column chunks).
  logic [15:0] vbias_row_w;
  logic [15:0] vacc_row_w;
  logic [15:0] vacc_c4_w;
  assign vbias_row_w = src2_off_q + {4'h0, bias_chunk_q};
  assign vacc_c4_w   = ({4'h0, bias_chunk_q} << 1) + {15'h0, c4_half_q};
  assign vacc_row_w  = src1_off_q +
                       ({1'b0, bias_row_idx_q} * {3'h0, n_chunks_i32_q}) +
                       vacc_c4_w;
  logic [15:0] vacc_dst_row_w;
  assign vacc_dst_row_w = dst_off_q +
                          ({1'b0, bias_row_idx_q} * {3'h0, n_chunks_i32_q}) +
                          vacc_c4_w;

  // VACC compute: 4 FP32 from ACCUM, 4 FP16 from the matching half of the
  // bias row, widen + add.
  logic [127:0] vacc_write_data_w;
  always_comb begin
    vacc_write_data_w = 128'h0;
    for (int lane = 0; lane < 4; lane++)
      vacc_write_data_w[(lane * 32) +: 32] = fp32_add_bits(
          get_u32(sram_a_rdata, lane),
          fp32_from_fp16_bits(
              get_u16(bias_data_q, (integer'(c4_half_q) << 2) + lane)));
  end

  // SCALE_MUL one-read paths.
  // For ABUF->ABUF: source row index = step_idx_q, 8 FP16 in, 8 FP16 out.
  // For ACCUM->ACCUM: source row index = step_idx_q, 4 FP32 in, 4 FP32 out.
  logic [15:0] sm1_src_row_w;
  logic [15:0] sm1_dst_row_w;
  assign sm1_src_row_w = src1_off_q + 16'(step_idx_q);
  assign sm1_dst_row_w = dst_off_q + 16'(step_idx_q);

  logic [127:0] sm1_write_data_w;
  always_comb begin
    sm1_write_data_w = 128'h0;
    if (src1_buf_q == BUF_ACCUM) begin
      for (int lane = 0; lane < 4; lane++)
        sm1_write_data_w[(lane * 32) +: 32] =
            fp32_mul_bits(get_u32(sram_b_rdata, lane), scale_fp32_q);
    end else begin
      for (int lane = 0; lane < 8; lane++)
        sm1_write_data_w[(lane * 16) +: 16] = fp32_to_fp16_bits(
            fp32_mul_bits(fp32_from_fp16_bits(get_u16(sram_b_rdata, lane)),
                          scale_fp32_q));
    end
  end

  // SCALE_MUL two-read path (ACCUM->ABUF narrowing).
  // step_idx_q iterates output FP16 rows; each output row consumes 2 ACCUM
  // rows (lanes 0..3 from row 0, lanes 4..7 from row 1).
  logic [15:0] sm2_src_row_w;
  logic [15:0] sm2_dst_row_w;
  assign sm2_src_row_w = src1_off_q + (16'(step_idx_q) << 1) + {15'h0, sm_part_q};
  assign sm2_dst_row_w = dst_off_q + 16'(step_idx_q);

  // Combine the two latched ACCUM rows into 8 FP16 lanes.
  logic [127:0] sm2_write_data_w;
  always_comb begin
    sm2_write_data_w = 128'h0;
    for (int lane = 0; lane < 4; lane++) begin
      sm2_write_data_w[(lane * 16) +: 16] = fp32_to_fp16_bits(
          fp32_mul_bits(get_u32(sm_row0_q, lane), scale_fp32_q));
      sm2_write_data_w[((lane + 4) * 16) +: 16] = fp32_to_fp16_bits(
          fp32_mul_bits(get_u32(sm_row1_q, lane), scale_fp32_q));
    end
  end

  // Main helper FSM. SRAM is synchronous, so read states issue a request and
  // the following state consumes the returned row.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state             <= H_IDLE;
      opcode_q          <= 5'h0;
      src1_buf_q        <= 2'b0;
      src2_buf_q        <= 2'b0;
      dst_buf_q         <= 2'b0;
      src1_off_q        <= 16'h0;
      src2_off_q        <= 16'h0;
      dst_off_q         <= 16'h0;
      sreg_q            <= 4'h0;
      b_length_q        <= 16'h0;
      b_src_rows_q      <= 6'h0;
      b_transpose_q     <= 1'b0;
      m_rows_q          <= 15'h0;
      n_tiles_q         <= 11'h0;
      n_fp16_chunks_q   <= 12'h0;
      n_chunks_i32_q    <= 13'h0;
      scale_fp32_q      <= 32'h0;
      fault_code_r      <= 4'(FAULT_NONE);
      step_idx_q        <= 32'h0;
      flat_backward_q   <= 1'b0;
      trans_row_count_q <= 16'h0;
      trans_cols_q      <= 16'h0;
      trans_elem_cols_q <= 16'h0;
      trans_rbase_q     <= 16'h0;
      trans_cbase_q     <= 16'h0;
      trans_height_q    <= 4'h0;
      trans_width_q     <= 4'h0;
      trans_src_row_idx_q <= 4'h0;
      trans_dst_row_idx_q <= 4'h0;
      bias_chunk_q      <= 12'h0;
      c4_half_q         <= 1'b0;
      bias_row_idx_q    <= 15'h0;
      bias_data_q       <= 128'h0;
      sm_part_q         <= 1'b0;
      sm_row0_q         <= 128'h0;
      sm_row1_q         <= 128'h0;
      for (int j = 0; j < 8; j++)
        trans_scratch_q[j] <= 128'h0;
    end else begin
      case (state)
        H_IDLE: begin
          if (dispatch) begin
            opcode_q        <= opcode;
            src1_buf_q      <= src1_buf;
            src2_buf_q      <= src2_buf;
            dst_buf_q       <= dst_buf;
            src1_off_q      <= src1_off;
            src2_off_q      <= src2_off;
            dst_off_q       <= dst_off;
            sreg_q          <= sreg;
            b_length_q      <= b_length;
            b_src_rows_q    <= b_src_rows;
            b_transpose_q   <= b_transpose;
            m_rows_q        <= dispatch_m_rows_w;
            n_tiles_q       <= dispatch_n_tiles_w;
            n_fp16_chunks_q <= dispatch_n_fp16_chunks_w;
            n_chunks_i32_q  <= dispatch_n_chunks_i32_w;
            scale_fp32_q    <= fp32_from_fp16_bits(scale0_data);

            if (dispatch_unsupported_w) begin
              fault_code_r <= 4'(FAULT_UNSUPPORTED_OP);
              state        <= H_FAULT;
            end else if (dispatch_sram_oob_w) begin
              fault_code_r <= 4'(FAULT_SRAM_OOB);
              state        <= H_FAULT;
            end else begin
              case (opcode)
                OP_BUF_COPY: begin
                  if (b_length == 16'h0) begin
                    state <= H_IDLE;
                  end else if (b_transpose) begin
                    trans_row_count_q   <= dispatch_src_rows_w;
                    trans_cols_q        <= dispatch_trans_elem_cols_w;
                    trans_elem_cols_q   <= dispatch_trans_elem_cols_w;
                    trans_rbase_q       <= 16'h0;
                    trans_cbase_q       <= 16'h0;
                    trans_height_q      <= 4'd8;
                    trans_width_q       <= 4'd8;
                    trans_src_row_idx_q <= 4'h0;
                    trans_dst_row_idx_q <= 4'h0;
                    state               <= H_TSRC_REQ;
                  end else begin
                    step_idx_q      <= 32'h0;
                    flat_backward_q <= dispatch_same_buf_overlap_w && (dst_off > src1_off);
                    state           <= H_FLAT_READ;
                  end
                end

                OP_VADD: begin
                  if (dispatch_is_vadd_fp16_w) begin
                    step_idx_q <= 32'h0;
                    state      <= H_V16_READ;
                  end else begin
                    bias_chunk_q   <= 12'h0;
                    c4_half_q      <= 1'b0;
                    bias_row_idx_q <= 15'h0;
                    state          <= H_VB_REQ;
                  end
                end

                OP_SCALE_MUL: begin
                  step_idx_q <= 32'h0;
                  if (dispatch_is_sm_acc2fp16_w) begin
                    sm_part_q <= 1'b0;
                    state     <= H_SM2_REQ;
                  end else begin
                    state <= H_SM1_REQ;
                  end
                end

                default: begin
                  fault_code_r <= 4'(FAULT_UNSUPPORTED_OP);
                  state        <= H_FAULT;
                end
              endcase
            end
          end
        end

        // BUF_COPY flat byte stream.
        H_FLAT_READ: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else begin
            state <= H_FLAT_WRITE;
          end
        end

        H_FLAT_WRITE: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else if (step_idx_q + 32'd1 >= {16'h0, b_length_q}) begin
            state <= H_IDLE;
          end else begin
            step_idx_q <= step_idx_q + 32'd1;
            state      <= H_FLAT_READ;
          end
        end

        // BUF_COPY transpose: 8 FP16 source rows -> 8 FP16 dest rows per block.
        H_TSRC_REQ: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else begin
            state <= H_TSRC_LATCH;
          end
        end

        H_TSRC_LATCH: begin
          trans_scratch_q[trans_src_row_idx_q[2:0]] <= sram_a_rdata;
          if (trans_src_row_idx_q + 4'd1 >= trans_height_q) begin
            trans_dst_row_idx_q <= 4'h0;
            state               <= H_TDST_WRITE;
          end else begin
            trans_src_row_idx_q <= trans_src_row_idx_q + 4'd1;
            state               <= H_TSRC_REQ;
          end
        end

        H_TDST_WRITE: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else if (trans_dst_row_idx_q + 4'd1 < trans_width_q) begin
            trans_dst_row_idx_q <= trans_dst_row_idx_q + 4'd1;
            state               <= H_TDST_WRITE;
          end else if ({16'h0, trans_cbase_q} + {28'h0, trans_width_q} < {16'h0, trans_cols_q}) begin
            trans_cbase_q       <= trans_cbase_q + 16'd8;
            trans_src_row_idx_q <= 4'h0;
            state               <= H_TSRC_REQ;
          end else if ({16'h0, trans_rbase_q} + {28'h0, trans_height_q} < {16'h0, trans_row_count_q}) begin
            trans_rbase_q       <= trans_rbase_q + 16'd8;
            trans_cbase_q       <= 16'h0;
            trans_src_row_idx_q <= 4'h0;
            state               <= H_TSRC_REQ;
          end else begin
            state <= H_IDLE;
          end
        end

        // FP16 ABUF VADD: read src1 (port B) + src2 (port A), widen + add +
        // narrow, write dst (port A).
        H_V16_READ: begin
          if (sram_a_fault || sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else begin
            state <= H_V16_WRITE;
          end
        end

        H_V16_WRITE: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else if (step_idx_q + 32'd1 >= dispatch_fp16_units_w) begin
            state <= H_IDLE;
          end else begin
            step_idx_q <= step_idx_q + 32'd1;
            state      <= H_V16_READ;
          end
        end

        // ACCUM VADD bias: load one FP16 bias row (8 columns) and reuse it
        // across all M rows for the 2 ACCUM 4-column chunks it covers.
        H_VB_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else begin
            state <= H_VB_LATCH;
          end
        end

        H_VB_LATCH: begin
          bias_data_q    <= sram_b_rdata;
          bias_row_idx_q <= 15'h0;
          c4_half_q      <= 1'b0;
          state          <= H_VACC_READ;
        end

        H_VACC_READ: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else begin
            state <= H_VACC_WRITE;
          end
        end

        H_VACC_WRITE: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else if (c4_half_q == 1'b0) begin
            // Advance to the high half of the same FP16 bias row, same M row.
            c4_half_q <= 1'b1;
            state     <= H_VACC_READ;
          end else if (bias_row_idx_q + 15'd1 < m_rows_q) begin
            bias_row_idx_q <= bias_row_idx_q + 15'd1;
            c4_half_q      <= 1'b0;
            state          <= H_VACC_READ;
          end else if ({20'h0, bias_chunk_q} + 32'd1 < {20'h0, n_fp16_chunks_q}) begin
            bias_chunk_q <= bias_chunk_q + 12'd1;
            state        <= H_VB_REQ;
          end else begin
            state <= H_IDLE;
          end
        end

        // SCALE_MUL one-read-one-write (ABUF->ABUF or ACCUM->ACCUM).
        H_SM1_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else begin
            state <= H_SM1_WRITE;
          end
        end

        H_SM1_WRITE: begin
          logic [31:0] total_rows_w;
          total_rows_w = (src1_buf_q == BUF_ACCUM) ? dispatch_fp32_units_w
                                                   : dispatch_fp16_units_w;
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else if (step_idx_q + 32'd1 >= total_rows_w) begin
            state <= H_IDLE;
          end else begin
            step_idx_q <= step_idx_q + 32'd1;
            state      <= H_SM1_REQ;
          end
        end

        // SCALE_MUL two-read-one-write (ACCUM -> ABUF narrow).
        H_SM2_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else begin
            state <= H_SM2_LATCH;
          end
        end

        H_SM2_LATCH: begin
          if (sm_part_q == 1'b0) begin
            sm_row0_q <= sram_b_rdata;
            sm_part_q <= 1'b1;
            state     <= H_SM2_REQ;
          end else begin
            sm_row1_q <= sram_b_rdata;
            state     <= H_SM2_WRITE;
          end
        end

        H_SM2_WRITE: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= H_FAULT;
          end else if (step_idx_q + 32'd1 >= dispatch_fp16_units_w) begin
            state <= H_IDLE;
          end else begin
            step_idx_q <= step_idx_q + 32'd1;
            sm_part_q  <= 1'b0;
            state      <= H_SM2_REQ;
          end
        end

        H_FAULT: ;

        default:
          state <= H_IDLE;
      endcase
    end
  end

  // State-to-SRAM decode.
  // Port A handles all writes and most reads; port B is used for the second
  // source stream in VADD and for SCALE_MUL reads.
  always_comb begin
    helper_busy       = (state != H_IDLE) && (state != H_FAULT);
    helper_fault      = (state == H_FAULT);
    helper_fault_code = fault_code_r;

    sram_a_en    = 1'b0;
    sram_a_we    = 1'b0;
    sram_a_buf   = src1_buf_q;
    sram_a_row   = 16'h0;
    sram_a_wdata = 128'h0;

    sram_b_en    = 1'b0;
    sram_b_buf   = src1_buf_q;
    sram_b_row   = 16'h0;

    case (state)
      H_FLAT_READ: begin
        sram_a_en  = 1'b1;
        sram_a_we  = 1'b0;
        sram_a_buf = src1_buf_q;
        sram_a_row = flat_src_row_w;
      end

      H_FLAT_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = flat_dst_row_w;
        sram_a_wdata = sram_a_rdata;
      end

      H_TSRC_REQ: begin
        sram_a_en  = 1'b1;
        sram_a_we  = 1'b0;
        sram_a_buf = src1_buf_q;
        sram_a_row = trans_src_row_w;
      end

      H_TDST_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = trans_dst_row_w;
        sram_a_wdata = trans_dst_data_w;
      end

      H_V16_READ: begin
        sram_a_en  = 1'b1;
        sram_a_we  = 1'b0;
        sram_a_buf = src2_buf_q;
        sram_a_row = v16_src2_row_w;
        sram_b_en  = 1'b1;
        sram_b_buf = src1_buf_q;
        sram_b_row = v16_src1_row_w;
      end

      H_V16_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = v16_dst_row_w;
        sram_a_wdata = v16_write_data_w;
      end

      H_VB_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src2_buf_q;
        sram_b_row = vbias_row_w;
      end

      H_VACC_READ: begin
        sram_a_en  = 1'b1;
        sram_a_we  = 1'b0;
        sram_a_buf = src1_buf_q;
        sram_a_row = vacc_row_w;
      end

      H_VACC_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = vacc_dst_row_w;
        sram_a_wdata = vacc_write_data_w;
      end

      H_SM1_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src1_buf_q;
        sram_b_row = sm1_src_row_w;
      end

      H_SM1_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = sm1_dst_row_w;
        sram_a_wdata = sm1_write_data_w;
      end

      H_SM2_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src1_buf_q;
        sram_b_row = sm2_src_row_w;
      end

      H_SM2_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = sm2_dst_row_w;
        sram_a_wdata = sm2_write_data_w;
      end

      default: ;
    endcase
  end

  // Unused — kept for synthesis-side lint quietness on byte helpers that may
  // become useful for diagnostic instrumentation.
  /* verilator lint_off UNUSED */
  logic [7:0] _unused_byte;
  assign _unused_byte = get_byte(128'h0, 0);
  /* verilator lint_on UNUSED */

endmodule

`endif // BLOCKING_HELPER_ENGINE_SV
