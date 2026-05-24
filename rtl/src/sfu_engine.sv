// Special-function-unit engine for W8A16 (Stage D rewrite).
//
// Supported operations (W8A16 contract):
//   - SOFTMAX   : FP16 ABUF or FP32 ACCUM input, FP16 ABUF output, row-wise.
//   - LAYERNORM : FP16 ABUF input, WBUF FP16 gamma/beta, FP16 ABUF output.
//   - GELU      : FP16 ABUF or FP32 ACCUM input, FP16 ABUF output.
//
// Removed in W8A16 (was W8A8):
//   - SOFTMAX_ATTNV (opcode OP_SOFTMAX_ATTNV) — the fused INT8 attention path
//     no longer matches the W8A16 software contract (sfu_w8a16 has no fused
//     codepath). The decode unit will raise FAULT_UNSUPPORTED_OP for this
//     opcode in Phase 5; until then the SFU dispatch validation handles it
//     here via the case-default fall-through that sets FAULT_UNSUPPORTED_OP.
//
// Datapath:
//   - ABUF rows hold 8 FP16 elements per 128-bit row (was 16 INT8). Reads
//     widen via fp32_from_fp16_bits, writes narrow via fp32_to_fp16_bits.
//   - ACCUM rows hold 4 FP32 elements per 128-bit row (same 4 × 32-bit
//     slots as the old INT32 ACCUM; bit interpretation flipped to FP32).
//     No scale multiplication on read/write.
//   - All reductions, exp/erf/gelu, mean/var/sqrt stay FP32-internal via
//     fp32_prim_pkg primitives — identical semantics to the W8A8 SFU's
//     internal math; only the endpoints flipped.
//   - Scale registers are still latched (SET_SCALE remains valid ISA) but
//     no longer participate in any arithmetic. They survive as dead state
//     for forward compatibility.
//
// Architectural contract:
//   - dispatched asynchronously through sfu_dispatch / sfu_busy
//   - serialized against DMA / helper / systolic at control level
//   - faults propagate asynchronously through sfu_fault / sfu_fault_code

`ifndef SFU_ENGINE_SV
`define SFU_ENGINE_SV

`include "taccel_pkg.sv"

module sfu_engine
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
  input  logic [9:0]   tile_m,
  input  logic [9:0]   tile_n,
  input  logic [9:0]   tile_k,
  input  logic [15:0]  scale0_data,
  input  logic [15:0]  scale1_data,
  input  logic [15:0]  scale2_data,
  input  logic [15:0]  scale3_data,

  output logic         sfu_busy,
  output logic         sfu_fault,
  output logic [3:0]   sfu_fault_code,

  output logic         sram_a_en,
  output logic         sram_a_we,
  output logic [1:0]   sram_a_buf,
  output logic [15:0]  sram_a_row,
  output logic [127:0] sram_a_wdata,
  input  logic         sram_a_fault,

  output logic         sram_b_en,
  output logic [1:0]   sram_b_buf,
  output logic [15:0]  sram_b_row,
  input  logic [127:0] sram_b_rdata,
  input  logic         sram_b_fault
);

  localparam int    SFU_MAX_ROW_ELEMS = 208;
  localparam fp32_t LN_EPS = 32'h358637BD;        // fp32(1e-6); see ARITH_CONTRACT.md

  typedef enum logic [4:0] {
    F_IDLE            = 5'd0,
    F_LN_PARAM_REQ    = 5'd1,
    F_LN_PARAM_LATCH  = 5'd2,
    F_ROW_FP16_REQ    = 5'd3,
    F_ROW_FP16_LATCH  = 5'd4,
    F_ROW_FP32_REQ    = 5'd5,
    F_ROW_FP32_LATCH  = 5'd6,
    F_ROW_COMPUTE     = 5'd7,
    F_ROW_PACK        = 5'd8,
    F_ROW_WRITE       = 5'd9,
    F_GELU_FP16_REQ   = 5'd10,
    F_GELU_FP16_LATCH = 5'd11,
    F_GELU_FP16_WRITE = 5'd12,
    F_GELU_FP32_REQ   = 5'd13,
    F_GELU_FP32_LATCH = 5'd14,
    F_GELU_FP32_WRITE = 5'd15,
    F_FAULT           = 5'd16
  } sfu_state_t;

  sfu_state_t state;

  logic [4:0]   opcode_q;
  logic [1:0]   src1_buf_q, src2_buf_q, dst_buf_q;
  logic [15:0]  src1_off_q, src2_off_q, dst_off_q;
  logic [3:0]   sreg_q;
  logic [14:0]  m_rows_q;
  logic [10:0]  n_tiles_q;
  logic [10:0]  k_tiles_q;
  // n_fp16_chunks_q = n_tiles_q * 2: number of FP16 SRAM rows per logical
  // row of n_elems_q elements (each FP16 row holds 8 elements).
  logic [11:0]  n_fp16_chunks_q;
  logic [12:0]  n_chunks_i32_q;
  logic [15:0]  n_elems_q;
  logic [15:0]  k_elems_q;
  logic [15:0]  ln_gamma_rows_q;
  logic [15:0]  ln_param_rows_q;
  logic [3:0]   fault_code_r;

  logic [14:0]  row_idx_q;
  logic [12:0]  read_idx_q;
  logic [11:0]  write_chunk_q;
  logic [0:0]   gelu_part_q;

  logic [127:0] gelu_row_fp16_q;
  logic [127:0] gelu_row0_q, gelu_row1_q;

  // Scale registers — latched on dispatch but UNUSED in W8A16 math. The
  // SET_SCALE opcode still writes the underlying register file; keeping
  // the FFs here avoids touching taccel_top wiring. May be deleted once
  // the SET_SCALE opcode itself is repurposed or removed.
  /* verilator lint_off UNUSED */
  fp32_t scale0_q, scale1_q, scale2_q, scale3_q;
  /* verilator lint_on UNUSED */

  fp32_t row_data_q   [0:SFU_MAX_ROW_ELEMS-1] /* verilator public_flat_rd */;
  fp32_t gamma_q      [0:SFU_MAX_ROW_ELEMS-1] /* verilator public_flat_rd */;
  fp32_t beta_q       [0:SFU_MAX_ROW_ELEMS-1] /* verilator public_flat_rd */;
  // FP16 bit patterns of the row's quantized output (8 per output SRAM row).
  logic [15:0] out_words_q [0:SFU_MAX_ROW_ELEMS-1] /* verilator public_flat_rd */;
  fp32_t ln_debug_mean_q /* verilator public_flat_rd */;
  fp32_t ln_debug_var_q /* verilator public_flat_rd */;
  fp32_t ln_debug_denom_q /* verilator public_flat_rd */;
  fp32_t ln_debug_y_q [0:15] /* verilator public_flat_rd */;

  logic [14:0] dispatch_m_rows_w;
  logic [10:0] dispatch_n_tiles_w;
  logic [10:0] dispatch_k_tiles_w;
  logic [11:0] dispatch_n_fp16_chunks_w;
  logic [12:0] dispatch_n_chunks_i32_w;
  logic [15:0] dispatch_n_elems_w;
  logic [15:0] dispatch_k_elems_w;
  logic [15:0] dispatch_ln_gamma_rows_w;
  logic [15:0] dispatch_ln_param_rows_w;
  logic [15:0] dispatch_src1_rows_w;
  logic [15:0] dispatch_src2_rows_w;
  logic [15:0] dispatch_dst_rows_w;
  logic        dispatch_softmax_accum_w;
  logic        dispatch_softmax_fp16_w;
  logic        dispatch_layernorm_w;
  logic        dispatch_gelu_accum_w;
  logic        dispatch_gelu_fp16_w;
  logic        dispatch_unsupported_w;
  logic        dispatch_sram_oob_w;

  logic [31:0] dispatch_src1_need_rows_w;
  logic [31:0] dispatch_src2_need_rows_w;
  logic [31:0] dispatch_dst_need_rows_w;

  logic [31:0] row_fp16_addr_w;
  logic [31:0] row_fp32_addr_w;
  logic [31:0] row_dst_addr_w;
  logic [31:0] ln_param_addr_w;
  logic [31:0] gelu_fp16_addr_w;
  logic [31:0] gelu_acc_addr_w;
  logic [31:0] gelu_dst_addr_w;

  logic [127:0] row_write_data_w;
  logic [127:0] row_write_q;
  logic [127:0] gelu_fp16_write_data_w;
  logic [127:0] gelu_fp32_write_data_w;

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

  assign dispatch_m_rows_w        = ({5'h0, tile_m} + 15'd1) << 4;
  assign dispatch_n_tiles_w       = {1'b0, tile_n} + 11'd1;
  assign dispatch_k_tiles_w       = {1'b0, tile_k} + 11'd1;
  assign dispatch_n_fp16_chunks_w = {1'b0, dispatch_n_tiles_w} << 1;
  assign dispatch_n_chunks_i32_w  = {2'h0, dispatch_n_tiles_w} << 2;
  assign dispatch_n_elems_w       = {1'b0, dispatch_n_tiles_w, 4'h0};
  assign dispatch_k_elems_w       = {1'b0, dispatch_k_tiles_w, 4'h0};
  assign dispatch_ln_gamma_rows_w = ({5'h0, dispatch_n_tiles_w}) << 1;
  assign dispatch_ln_param_rows_w = ({5'h0, dispatch_n_tiles_w}) << 2;
  assign dispatch_src1_rows_w     = buf_rows(src1_buf);
  assign dispatch_src2_rows_w     = buf_rows(src2_buf);
  assign dispatch_dst_rows_w      = buf_rows(dst_buf);

  assign dispatch_softmax_accum_w = (opcode == OP_SOFTMAX) &&
                                    (src1_buf == BUF_ACCUM) &&
                                    (dst_buf != BUF_ACCUM);
  assign dispatch_softmax_fp16_w  = (opcode == OP_SOFTMAX) &&
                                    (src1_buf != BUF_ACCUM) &&
                                    (dst_buf != BUF_ACCUM);
  assign dispatch_layernorm_w     = (opcode == OP_LAYERNORM) &&
                                    (src1_buf == BUF_ABUF) &&
                                    (src2_buf == BUF_WBUF) &&
                                    (dst_buf != BUF_ACCUM);
  assign dispatch_gelu_accum_w    = (opcode == OP_GELU) &&
                                    (src1_buf == BUF_ACCUM) &&
                                    (dst_buf != BUF_ACCUM);
  assign dispatch_gelu_fp16_w     = (opcode == OP_GELU) &&
                                    (src1_buf == BUF_ABUF) &&
                                    (dst_buf != BUF_ACCUM);

  always_comb begin
    dispatch_unsupported_w = 1'b0;
    dispatch_sram_oob_w    = 1'b0;
    dispatch_src1_need_rows_w = 32'd0;
    dispatch_src2_need_rows_w = 32'd0;
    dispatch_dst_need_rows_w  = 32'd0;

    case (opcode)
      OP_SOFTMAX: begin
        if (sreg == 4'hF)
          dispatch_unsupported_w = 1'b1;
        if (!(dispatch_softmax_accum_w || dispatch_softmax_fp16_w))
          dispatch_unsupported_w = 1'b1;
        if (integer'(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
          dispatch_unsupported_w = 1'b1;

        if (dispatch_softmax_accum_w)
          dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_n_chunks_i32_w;
        else if (dispatch_softmax_fp16_w)
          dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_n_fp16_chunks_w;
        dispatch_dst_need_rows_w = dispatch_m_rows_w * dispatch_n_fp16_chunks_w;
      end

      OP_LAYERNORM: begin
        if (sreg == 4'hF)
          dispatch_unsupported_w = 1'b1;
        if (!dispatch_layernorm_w)
          dispatch_unsupported_w = 1'b1;
        if (integer'(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
          dispatch_unsupported_w = 1'b1;

        dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_n_fp16_chunks_w;
        dispatch_src2_need_rows_w = {16'h0, dispatch_ln_param_rows_w};
        dispatch_dst_need_rows_w  = dispatch_m_rows_w * dispatch_n_fp16_chunks_w;
      end

      OP_GELU: begin
        if (sreg == 4'hF)
          dispatch_unsupported_w = 1'b1;
        if (!(dispatch_gelu_accum_w || dispatch_gelu_fp16_w))
          dispatch_unsupported_w = 1'b1;

        if (dispatch_gelu_accum_w)
          dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_n_chunks_i32_w;
        else if (dispatch_gelu_fp16_w)
          dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_n_fp16_chunks_w;
        dispatch_dst_need_rows_w = dispatch_m_rows_w * dispatch_n_fp16_chunks_w;
      end

      default:
        dispatch_unsupported_w = 1'b1;
    endcase

    dispatch_sram_oob_w =
        ({16'h0, src1_off} + dispatch_src1_need_rows_w > {16'h0, dispatch_src1_rows_w}) ||
        ({16'h0, src2_off} + dispatch_src2_need_rows_w > {16'h0, dispatch_src2_rows_w}) ||
        ({16'h0, dst_off}  + dispatch_dst_need_rows_w  > {16'h0, dispatch_dst_rows_w});
  end

  // Address calculations. All ABUF-side rows are FP16 (8 elements per 128-bit
  // row), so the per-row stride is n_fp16_chunks_q. ACCUM rows are FP32 (4
  // elements per row), stride n_chunks_i32_q.
  assign row_fp16_addr_w = {16'h0, src1_off_q} +
                           ({17'h0, row_idx_q} * {20'h0, n_fp16_chunks_q}) +
                           {19'h0, read_idx_q};
  assign row_fp32_addr_w = {16'h0, src1_off_q} +
                           ({17'h0, row_idx_q} * {19'h0, n_chunks_i32_q}) +
                           {19'h0, read_idx_q};
  assign row_dst_addr_w  = {16'h0, dst_off_q} +
                           ({17'h0, row_idx_q} * {20'h0, n_fp16_chunks_q}) +
                           {20'h0, write_chunk_q};
  assign ln_param_addr_w = {16'h0, src2_off_q} + {19'h0, read_idx_q};
  assign gelu_fp16_addr_w = {16'h0, src1_off_q} +
                            ({17'h0, row_idx_q} * {20'h0, n_fp16_chunks_q}) +
                            {20'h0, write_chunk_q};
  // For GELU FP32->FP16, each FP16 output row consumes 2 ACCUM rows
  // (write_chunk_q * 2 + gelu_part_q).
  assign gelu_acc_addr_w = {16'h0, src1_off_q} +
                           ({17'h0, row_idx_q} * {19'h0, n_chunks_i32_q}) +
                           ({20'h0, write_chunk_q} << 1) +
                           {31'h0, gelu_part_q};
  assign gelu_dst_addr_w = {16'h0, dst_off_q} +
                           ({17'h0, row_idx_q} * {20'h0, n_fp16_chunks_q}) +
                           {20'h0, write_chunk_q};

  always_comb begin
    row_write_data_w        = 128'h0;
    gelu_fp16_write_data_w  = 128'h0;
    gelu_fp32_write_data_w  = 128'h0;

    for (int lane = 0; lane < 8; lane++) begin
      int    idx;
      fp32_t x_b;
      idx = integer'(write_chunk_q) * 8 + lane;
      if (idx < integer'(n_elems_q))
        row_write_data_w[(lane * 16) +: 16] = out_words_q[idx];

      // GELU on FP16 input: read 8 FP16 from gelu_row_fp16_q, write 8 FP16.
      x_b = fp32_from_fp16_bits(get_u16(gelu_row_fp16_q, lane));
      gelu_fp16_write_data_w[(lane * 16) +: 16] = fp32_to_fp16_bits(fp32_gelu_bits(x_b));

      // GELU on FP32 ACCUM: 8 elements per output FP16 row come from
      // 2 ACCUM rows (4 FP32 each): lanes 0..3 from gelu_row0_q, 4..7
      // from gelu_row1_q.
      if (lane < 4) begin
        x_b = get_u32(gelu_row0_q, lane);
        gelu_fp32_write_data_w[(lane * 16) +: 16] = fp32_to_fp16_bits(fp32_gelu_bits(x_b));
        x_b = get_u32(gelu_row1_q, lane);
        gelu_fp32_write_data_w[((lane + 4) * 16) +: 16] = fp32_to_fp16_bits(fp32_gelu_bits(x_b));
      end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state          <= F_IDLE;
      opcode_q       <= 5'h0;
      src1_buf_q     <= 2'b0;
      src2_buf_q     <= 2'b0;
      dst_buf_q      <= 2'b0;
      src1_off_q     <= 16'h0;
      src2_off_q     <= 16'h0;
      dst_off_q      <= 16'h0;
      sreg_q         <= 4'h0;
      m_rows_q       <= 15'h0;
      n_tiles_q      <= 11'h0;
      k_tiles_q      <= 11'h0;
      n_fp16_chunks_q<= 12'h0;
      n_chunks_i32_q <= 13'h0;
      n_elems_q      <= 16'h0;
      k_elems_q      <= 16'h0;
      ln_gamma_rows_q<= 16'h0;
      ln_param_rows_q<= 16'h0;
      fault_code_r   <= 4'(FAULT_NONE);
      row_idx_q      <= 15'h0;
      read_idx_q     <= 13'h0;
      write_chunk_q  <= 12'h0;
      gelu_part_q    <= 1'h0;
      gelu_row_fp16_q<= 128'h0;
      gelu_row0_q    <= 128'h0;
      gelu_row1_q    <= 128'h0;
      row_write_q    <= 128'h0;
      scale0_q       <= 32'd0;
      scale1_q       <= 32'd0;
      scale2_q       <= 32'd0;
      scale3_q       <= 32'd0;
      ln_debug_mean_q <= 32'd0;
      ln_debug_var_q <= 32'd0;
      ln_debug_denom_q <= 32'd0;
      for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
        row_data_q[i] <= 32'd0;
        gamma_q[i]    <= 32'd0;
        beta_q[i]     <= 32'd0;
        out_words_q[i] <= 16'h0000;
      end
      for (int i = 0; i < 16; i++)
        ln_debug_y_q[i] <= 32'd0;
    end else begin
      case (state)
        F_IDLE: begin
          if (dispatch) begin
            opcode_q        <= opcode;
            src1_buf_q      <= src1_buf;
            src2_buf_q      <= src2_buf;
            dst_buf_q       <= dst_buf;
            src1_off_q      <= src1_off;
            src2_off_q      <= src2_off;
            dst_off_q       <= dst_off;
            sreg_q          <= sreg;
            m_rows_q        <= dispatch_m_rows_w;
            n_tiles_q       <= dispatch_n_tiles_w;
            k_tiles_q       <= dispatch_k_tiles_w;
            n_fp16_chunks_q <= dispatch_n_fp16_chunks_w;
            n_chunks_i32_q  <= dispatch_n_chunks_i32_w;
            n_elems_q       <= dispatch_n_elems_w;
            k_elems_q       <= dispatch_k_elems_w;
            ln_gamma_rows_q <= dispatch_ln_gamma_rows_w;
            ln_param_rows_q <= dispatch_ln_param_rows_w;
            scale0_q        <= fp32_from_fp16_bits(scale0_data);
            scale1_q        <= fp32_from_fp16_bits(scale1_data);
            scale2_q        <= fp32_from_fp16_bits(scale2_data);
            scale3_q        <= fp32_from_fp16_bits(scale3_data);
            ln_debug_mean_q <= 32'd0;
            ln_debug_var_q <= 32'd0;
            ln_debug_denom_q <= 32'd0;
            for (int i = 0; i < 16; i++)
              ln_debug_y_q[i] <= 32'd0;
            row_idx_q       <= 15'h0;
            read_idx_q      <= 13'h0;
            write_chunk_q   <= 12'h0;
            gelu_part_q     <= 1'h0;

            if (dispatch_unsupported_w) begin
              fault_code_r <= 4'(FAULT_UNSUPPORTED_OP);
              state        <= F_FAULT;
            end else if (dispatch_sram_oob_w) begin
              fault_code_r <= 4'(FAULT_SRAM_OOB);
              state        <= F_FAULT;
            end else begin
              case (opcode)
                OP_SOFTMAX: begin
                  if (src1_buf == BUF_ACCUM)
                    state <= F_ROW_FP32_REQ;
                  else
                    state <= F_ROW_FP16_REQ;
                end

                OP_LAYERNORM:
                  state <= F_LN_PARAM_REQ;

                OP_GELU: begin
                  if (src1_buf == BUF_ACCUM)
                    state <= F_GELU_FP32_REQ;
                  else
                    state <= F_GELU_FP16_REQ;
                end

                default: begin
                  fault_code_r <= 4'(FAULT_UNSUPPORTED_OP);
                  state        <= F_FAULT;
                end
              endcase
            end
          end
        end

        F_LN_PARAM_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else begin
            state <= F_LN_PARAM_LATCH;
          end
        end

        F_LN_PARAM_LATCH: begin
          integer base_idx;
          base_idx = (integer'(read_idx_q) < integer'(ln_gamma_rows_q)) ?
                     (integer'(read_idx_q) * 8) :
                     ((integer'(read_idx_q) - integer'(ln_gamma_rows_q)) * 8);
          for (int lane = 0; lane < 8; lane++) begin
            if ((base_idx + lane) < integer'(n_elems_q)) begin
              if (integer'(read_idx_q) < integer'(ln_gamma_rows_q))
                gamma_q[base_idx + lane] <= fp32_from_fp16_bits(get_u16(sram_b_rdata, lane));
              else
                beta_q[base_idx + lane] <= fp32_from_fp16_bits(get_u16(sram_b_rdata, lane));
            end
          end

          if ((integer'(read_idx_q) + 1) < integer'(ln_param_rows_q)) begin
            read_idx_q <= read_idx_q + 13'd1;
            state      <= F_LN_PARAM_REQ;
          end else begin
            read_idx_q <= 13'h0;
            state      <= F_ROW_FP16_REQ;
          end
        end

        F_ROW_FP16_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else begin
            state <= F_ROW_FP16_LATCH;
          end
        end

        F_ROW_FP16_LATCH: begin
          integer base_idx;
          base_idx = integer'(read_idx_q) * 8;
          for (int lane = 0; lane < 8; lane++) begin
            if ((base_idx + lane) < integer'(n_elems_q))
              row_data_q[base_idx + lane] <=
                  fp32_from_fp16_bits(get_u16(sram_b_rdata, lane));
          end

          if (read_idx_q + 13'd1 < {1'h0, n_fp16_chunks_q}) begin
            read_idx_q <= read_idx_q + 13'd1;
            state      <= F_ROW_FP16_REQ;
          end else begin
            write_chunk_q <= 12'h0;
            state         <= F_ROW_COMPUTE;
          end
        end

        F_ROW_FP32_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else begin
            state <= F_ROW_FP32_LATCH;
          end
        end

        F_ROW_FP32_LATCH: begin
          integer base_idx;
          base_idx = integer'(read_idx_q) * 4;
          for (int lane = 0; lane < 4; lane++) begin
            if ((base_idx + lane) < integer'(n_elems_q))
              row_data_q[base_idx + lane] <= get_u32(sram_b_rdata, lane);
          end

          if (read_idx_q + 13'd1 < n_chunks_i32_q) begin
            read_idx_q <= read_idx_q + 13'd1;
            state      <= F_ROW_FP32_REQ;
          end else begin
            write_chunk_q <= 12'h0;
            state         <= F_ROW_COMPUTE;
          end
        end

        F_ROW_COMPUTE: begin
          if (opcode_q == OP_SOFTMAX) begin
            fp32_t row_max_b;
            fp32_t exp_sum_b;
            fp32_t exp_b;
            row_max_b = row_data_q[0];
            for (int i = 1; i < SFU_MAX_ROW_ELEMS; i++) begin
              if ((i < integer'(n_elems_q)) && fp32_gt(row_data_q[i], row_max_b))
                row_max_b = row_data_q[i];
            end

            exp_sum_b = 32'd0;
            for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
              if (i < integer'(n_elems_q)) begin
                exp_b = fp32_exp_bits(fp32_sub_bits(row_data_q[i], row_max_b));
                exp_sum_b = fp32_add_bits(exp_sum_b, exp_b);
              end
            end

            for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
              if (i < integer'(n_elems_q)) begin
                exp_b = fp32_exp_bits(fp32_sub_bits(row_data_q[i], row_max_b));
                out_words_q[i] <= fp32_to_fp16_bits(fp32_div_bits(exp_b, exp_sum_b));
              end
            end
          end else begin
            fp32_t sum_b;
            fp32_t mean_b;
            fp32_t var_b;
            fp32_t denom_b;
            fp32_t n_b;
            n_b = fp32_from_i32({16'd0, n_elems_q});
            sum_b = 32'd0;
            for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
              if (i < integer'(n_elems_q))
                sum_b = fp32_add_bits(sum_b, row_data_q[i]);
            end
            mean_b = fp32_div_bits(sum_b, n_b);

            var_b = 32'd0;
            for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
              if (i < integer'(n_elems_q)) begin
                fp32_t diff_b;
                diff_b = fp32_sub_bits(row_data_q[i], mean_b);
                var_b = fp32_add_bits(var_b, fp32_mul_bits(diff_b, diff_b));
              end
            end
            var_b = fp32_div_bits(var_b, n_b);
            denom_b = fp32_sqrt_bits(fp32_add_bits(var_b, LN_EPS));
            ln_debug_mean_q <= mean_b;
            ln_debug_var_q <= var_b;
            ln_debug_denom_q <= denom_b;

            for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
              fp32_t y_b;
              if (i < integer'(n_elems_q)) begin
                y_b = fp32_add_bits(
                    fp32_mul_bits(
                        fp32_div_bits(fp32_sub_bits(row_data_q[i], mean_b), denom_b),
                        gamma_q[i]),
                    beta_q[i]);
                out_words_q[i] <= fp32_to_fp16_bits(y_b);
                if (i < 16)
                  ln_debug_y_q[i] <= y_b;
              end else if (i < 16) begin
                ln_debug_y_q[i] <= 32'd0;
              end
            end
          end
          state <= F_ROW_PACK;
        end

        F_ROW_PACK: begin
          row_write_q <= row_write_data_w;
          state <= F_ROW_WRITE;
        end

        F_ROW_WRITE: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else if (write_chunk_q + 12'd1 < {1'h0, n_fp16_chunks_q}) begin
            write_chunk_q <= write_chunk_q + 12'd1;
            state         <= F_ROW_PACK;
          end else if (row_idx_q + 15'd1 < m_rows_q) begin
            row_idx_q     <= row_idx_q + 15'd1;
            read_idx_q    <= 13'h0;
            write_chunk_q <= 12'h0;
            if ((opcode_q == OP_SOFTMAX) && (src1_buf_q == BUF_ACCUM))
              state <= F_ROW_FP32_REQ;
            else
              state <= F_ROW_FP16_REQ;
          end else begin
            state <= F_IDLE;
          end
        end

        F_GELU_FP16_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else begin
            state <= F_GELU_FP16_LATCH;
          end
        end

        F_GELU_FP16_LATCH: begin
          gelu_row_fp16_q <= sram_b_rdata;
          state           <= F_GELU_FP16_WRITE;
        end

        F_GELU_FP16_WRITE: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else if (write_chunk_q + 12'd1 < {1'h0, n_fp16_chunks_q}) begin
            write_chunk_q <= write_chunk_q + 12'd1;
            state         <= F_GELU_FP16_REQ;
          end else if (row_idx_q + 15'd1 < m_rows_q) begin
            row_idx_q     <= row_idx_q + 15'd1;
            write_chunk_q <= 12'h0;
            state         <= F_GELU_FP16_REQ;
          end else begin
            state <= F_IDLE;
          end
        end

        F_GELU_FP32_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else begin
            state <= F_GELU_FP32_LATCH;
          end
        end

        F_GELU_FP32_LATCH: begin
          if (gelu_part_q == 1'd0)
            gelu_row0_q <= sram_b_rdata;
          else
            gelu_row1_q <= sram_b_rdata;

          if (gelu_part_q == 1'd1) begin
            state <= F_GELU_FP32_WRITE;
          end else begin
            gelu_part_q <= 1'd1;
            state       <= F_GELU_FP32_REQ;
          end
        end

        F_GELU_FP32_WRITE: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else if (write_chunk_q + 12'd1 < {1'h0, n_fp16_chunks_q}) begin
            write_chunk_q <= write_chunk_q + 12'd1;
            gelu_part_q   <= 1'h0;
            state         <= F_GELU_FP32_REQ;
          end else if (row_idx_q + 15'd1 < m_rows_q) begin
            row_idx_q     <= row_idx_q + 15'd1;
            write_chunk_q <= 12'h0;
            gelu_part_q   <= 1'h0;
            state         <= F_GELU_FP32_REQ;
          end else begin
            state <= F_IDLE;
          end
        end

        F_FAULT: ;

        default:
          state <= F_IDLE;
      endcase
    end
  end

  always_comb begin
    sfu_busy       = (state != F_IDLE) && (state != F_FAULT);
    sfu_fault      = (state == F_FAULT);
    sfu_fault_code = fault_code_r;

    sram_a_en    = 1'b0;
    sram_a_we    = 1'b0;
    sram_a_buf   = dst_buf_q;
    sram_a_row   = 16'h0;
    sram_a_wdata = 128'h0;

    sram_b_en    = 1'b0;
    sram_b_buf   = src1_buf_q;
    sram_b_row   = 16'h0;

    case (state)
      F_LN_PARAM_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src2_buf_q;
        sram_b_row = ln_param_addr_w[15:0];
      end

      F_ROW_FP16_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src1_buf_q;
        sram_b_row = row_fp16_addr_w[15:0];
      end

      F_ROW_FP32_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src1_buf_q;
        sram_b_row = row_fp32_addr_w[15:0];
      end

      F_ROW_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = row_dst_addr_w[15:0];
        sram_a_wdata = row_write_q;
      end

      F_GELU_FP16_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src1_buf_q;
        sram_b_row = gelu_fp16_addr_w[15:0];
      end

      F_GELU_FP16_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = gelu_dst_addr_w[15:0];
        sram_a_wdata = gelu_fp16_write_data_w;
      end

      F_GELU_FP32_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src1_buf_q;
        sram_b_row = gelu_acc_addr_w[15:0];
      end

      F_GELU_FP32_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = gelu_dst_addr_w[15:0];
        sram_a_wdata = gelu_fp32_write_data_w;
      end

      default: ;
    endcase
  end

endmodule

`endif // SFU_ENGINE_SV
