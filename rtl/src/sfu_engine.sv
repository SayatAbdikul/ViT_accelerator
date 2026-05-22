// Special-function-unit engine for Stage D numerical parity.
//
// Supported operations:
//   - SOFTMAX   : INT8/INT32 input, INT8 output, row-wise across full logical N
//   - LAYERNORM : ABUF INT8 input, WBUF FP16 gamma/beta, INT8 output
//   - GELU      : ABUF INT8 or ACCUM INT32 input, INT8 output
//   - SOFTMAX_ATTNV : fused softmax(QK^T) @ V with INT8 output
//
// Architectural contract:
//   - dispatched asynchronously through sfu_dispatch / sfu_busy
//   - serialized against DMA / helper / systolic at control level in Stage D
//   - faults propagate asynchronously through sfu_fault / sfu_fault_code
//
// Implementation note:
//   Stage D prioritizes functional parity with the software golden model over
//   synthesis-oriented microarchitecture. The engine therefore uses real-valued
//   intermediate storage plus explicit FP32 rounding helpers to preserve the
//   architectural "all SFU internal operations use FP32" contract under the
//   current simulator, which does not model shortreal arithmetic directly.

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
    F_IDLE          = 5'd0,
    F_LN_PARAM_REQ  = 5'd1,
    F_LN_PARAM_LATCH= 5'd2,
    F_ROW_I8_REQ    = 5'd3,
    F_ROW_I8_LATCH  = 5'd4,
    F_ROW_I32_REQ   = 5'd5,
    F_ROW_I32_LATCH = 5'd6,
    F_ROW_COMPUTE   = 5'd7,
    F_ROW_PACK      = 5'd8,
    F_ROW_WRITE     = 5'd9,
    F_GELU_I8_REQ   = 5'd10,
    F_GELU_I8_LATCH = 5'd11,
    F_GELU_I8_WRITE = 5'd12,
    F_GELU_I32_REQ  = 5'd13,
    F_GELU_I32_LATCH= 5'd14,
    F_GELU_I32_WRITE= 5'd15,
    F_ATTN_QKT_REQ  = 5'd16,
    F_ATTN_QKT_LATCH= 5'd17,
    F_ATTN_PREP     = 5'd18,
    F_ATTN_V_REQ    = 5'd19,
    F_ATTN_V_LATCH  = 5'd20,
    F_ATTN_WRITE    = 5'd21,
    F_FAULT         = 5'd22
  } sfu_state_t;

  sfu_state_t state;

  logic [4:0]   opcode_q;
  logic [1:0]   src1_buf_q, src2_buf_q, dst_buf_q;
  logic [15:0]  src1_off_q, src2_off_q, dst_off_q;
  logic [3:0]   sreg_q;
  logic [14:0]  m_rows_q;
  logic [10:0]  n_tiles_q;
  logic [10:0]  k_tiles_q;
  logic [12:0]  n_chunks_i32_q;
  logic [12:0]  k_chunks_i32_q;
  logic [15:0]  n_elems_q;
  logic [15:0]  k_elems_q;
  logic [15:0]  ln_gamma_rows_q;
  logic [15:0]  ln_param_rows_q;
  logic [3:0]   fault_code_r;

  logic [14:0]  row_idx_q;
  logic [12:0]  read_idx_q;
  logic [10:0]  write_chunk_q;
  logic [1:0]   gelu_part_q;
  logic [15:0]  attn_k_idx_q;

  logic [127:0] gelu_i8_row_q;
  logic [127:0] gelu_row0_q, gelu_row1_q, gelu_row2_q, gelu_row3_q;

  fp32_t scale0_q /* verilator public_flat_rd */, scale1_q /* verilator public_flat_rd */,
         scale2_q /* verilator public_flat_rd */, scale3_q /* verilator public_flat_rd */;
  fp32_t row_data_q   [0:SFU_MAX_ROW_ELEMS-1] /* verilator public_flat_rd */;
  fp32_t attn_accum_q [0:SFU_MAX_ROW_ELEMS-1];
  fp32_t gamma_q      [0:SFU_MAX_ROW_ELEMS-1] /* verilator public_flat_rd */;
  fp32_t beta_q       [0:SFU_MAX_ROW_ELEMS-1] /* verilator public_flat_rd */;
  logic [7:0] out_bytes_q [0:SFU_MAX_ROW_ELEMS-1] /* verilator public_flat_rd */;
  fp32_t attn_row_max_q;
  fp32_t attn_exp_sum_q;
  fp32_t ln_debug_mean_q /* verilator public_flat_rd */;
  fp32_t ln_debug_var_q /* verilator public_flat_rd */;
  fp32_t ln_debug_denom_q /* verilator public_flat_rd */;
  fp32_t ln_debug_y_q [0:15] /* verilator public_flat_rd */;

  logic [14:0] dispatch_m_rows_w;
  logic [10:0] dispatch_n_tiles_w;
  logic [10:0] dispatch_k_tiles_w;
  logic [12:0] dispatch_n_chunks_i32_w;
  logic [12:0] dispatch_k_chunks_i32_w;
  logic [15:0] dispatch_n_elems_w;
  logic [15:0] dispatch_k_elems_w;
  logic [15:0] dispatch_ln_gamma_rows_w;
  logic [15:0] dispatch_ln_param_rows_w;
  logic [15:0] dispatch_src1_rows_w;
  logic [15:0] dispatch_src2_rows_w;
  logic [15:0] dispatch_dst_rows_w;
  logic        dispatch_softmax_accum_w;
  logic        dispatch_softmax_int8_w;
  logic        dispatch_layernorm_w;
  logic        dispatch_gelu_accum_w;
  logic        dispatch_gelu_int8_w;
  logic        dispatch_softmax_attnv_w;
  logic        dispatch_unsupported_w;
  logic        dispatch_sram_oob_w;

  logic [31:0] dispatch_src1_need_rows_w;
  logic [31:0] dispatch_src2_need_rows_w;
  logic [31:0] dispatch_dst_need_rows_w;

  logic [31:0] row_i8_addr_w;
  logic [31:0] row_i32_addr_w;
  logic [31:0] row_dst_addr_w;
  logic [31:0] ln_param_addr_w;
  logic [31:0] gelu_i8_addr_w;
  logic [31:0] gelu_acc_addr_w;
  logic [31:0] gelu_dst_addr_w;
  logic [31:0] attn_qkt_addr_w;
  logic [31:0] attn_v_addr_w;

  logic [127:0] row_write_data_w;
  logic [127:0] row_write_q;
  logic [127:0] gelu_i8_write_data_w;
  logic [127:0] gelu_i32_write_data_w;
  logic [127:0] attn_write_data_w;


  // FP16 widening and INT->FP32 conversions live in fp32_prim_pkg
  // (fp32_from_fp16_bits / fp32_from_i8 / fp32_from_i32). The legacy
  // pow2_int / fp16_to_real helpers were removed for synthesis.

  function automatic logic signed [7:0] get_i8(
    input logic [127:0] row,
    input integer       idx
  );
    begin
      get_i8 = row[(idx * 8) +: 8];
    end
  endfunction

  function automatic logic signed [31:0] get_i32(
    input logic [127:0] row,
    input integer       idx
  );
    begin
      get_i32 = row[(idx * 32) +: 32];
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

  function automatic logic [7:0] quantize_to_i8(
    input fp32_t value_bits,
    input fp32_t out_scale_bits
  );
    s64_t q;
    begin
      // fp32_quantize_i8_bits handles out_scale==0 and clamps to [-128,127].
      q = fp32_quantize_i8_bits(value_bits, out_scale_bits);
      quantize_to_i8 = q[7:0];
    end
  endfunction

  assign dispatch_m_rows_w        = ({5'h0, tile_m} + 15'd1) << 4;
  assign dispatch_n_tiles_w       = {1'b0, tile_n} + 11'd1;
  assign dispatch_k_tiles_w       = {1'b0, tile_k} + 11'd1;
  assign dispatch_n_chunks_i32_w  = dispatch_n_tiles_w << 2;
  assign dispatch_k_chunks_i32_w  = dispatch_k_tiles_w << 2;
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
  assign dispatch_softmax_int8_w  = (opcode == OP_SOFTMAX) &&
                                    (src1_buf != BUF_ACCUM) &&
                                    (dst_buf != BUF_ACCUM);
  assign dispatch_layernorm_w     = (opcode == OP_LAYERNORM) &&
                                    (src1_buf == BUF_ABUF) &&
                                    (src2_buf == BUF_WBUF) &&
                                    (dst_buf != BUF_ACCUM);
  assign dispatch_gelu_accum_w    = (opcode == OP_GELU) &&
                                    (src1_buf == BUF_ACCUM) &&
                                    (dst_buf != BUF_ACCUM);
  assign dispatch_gelu_int8_w     = (opcode == OP_GELU) &&
                                    (src1_buf == BUF_ABUF) &&
                                    (dst_buf != BUF_ACCUM);
  assign dispatch_softmax_attnv_w = (opcode == OP_SOFTMAX_ATTNV) &&
                                    (src1_buf == BUF_ACCUM) &&
                                    (src2_buf != BUF_ACCUM) &&
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
        if (!(dispatch_softmax_accum_w || dispatch_softmax_int8_w))
          dispatch_unsupported_w = 1'b1;
        if (integer'(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
          dispatch_unsupported_w = 1'b1;

        if (dispatch_softmax_accum_w)
          dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_n_chunks_i32_w;
        else if (dispatch_softmax_int8_w)
          dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_n_tiles_w;
        dispatch_dst_need_rows_w = dispatch_m_rows_w * dispatch_n_tiles_w;
      end

      OP_LAYERNORM: begin
        if (sreg == 4'hF)
          dispatch_unsupported_w = 1'b1;
        if (!dispatch_layernorm_w)
          dispatch_unsupported_w = 1'b1;
        if (integer'(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS)
          dispatch_unsupported_w = 1'b1;

        dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_n_tiles_w;
        dispatch_src2_need_rows_w = {16'h0, dispatch_ln_param_rows_w};
        dispatch_dst_need_rows_w  = dispatch_m_rows_w * dispatch_n_tiles_w;
      end

      OP_GELU: begin
        if (sreg == 4'hF)
          dispatch_unsupported_w = 1'b1;
        if (!(dispatch_gelu_accum_w || dispatch_gelu_int8_w))
          dispatch_unsupported_w = 1'b1;

        if (dispatch_gelu_accum_w)
          dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_n_chunks_i32_w;
        else if (dispatch_gelu_int8_w)
          dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_n_tiles_w;
        dispatch_dst_need_rows_w = dispatch_m_rows_w * dispatch_n_tiles_w;
      end

      OP_SOFTMAX_ATTNV: begin
        if (sreg > 4'd12)
          dispatch_unsupported_w = 1'b1;
        if (!dispatch_softmax_attnv_w)
          dispatch_unsupported_w = 1'b1;
        if ((integer'(dispatch_k_elems_w) > SFU_MAX_ROW_ELEMS) ||
            (integer'(dispatch_n_elems_w) > SFU_MAX_ROW_ELEMS))
          dispatch_unsupported_w = 1'b1;

        dispatch_src1_need_rows_w = dispatch_m_rows_w * dispatch_k_chunks_i32_w;
        dispatch_src2_need_rows_w = dispatch_k_elems_w * dispatch_n_tiles_w;
        dispatch_dst_need_rows_w  = dispatch_m_rows_w * dispatch_n_tiles_w;
      end

      default:
        dispatch_unsupported_w = 1'b1;
    endcase

    dispatch_sram_oob_w =
        ({16'h0, src1_off} + dispatch_src1_need_rows_w > {16'h0, dispatch_src1_rows_w}) ||
        ({16'h0, src2_off} + dispatch_src2_need_rows_w > {16'h0, dispatch_src2_rows_w}) ||
        ({16'h0, dst_off}  + dispatch_dst_need_rows_w  > {16'h0, dispatch_dst_rows_w});
  end

  assign row_i8_addr_w  = {16'h0, src1_off_q} +
                          ({17'h0, row_idx_q} * {21'h0, n_tiles_q}) +
                          {19'h0, read_idx_q};
  assign row_i32_addr_w = {16'h0, src1_off_q} +
                          ({17'h0, row_idx_q} * {19'h0, n_chunks_i32_q}) +
                          {19'h0, read_idx_q};
  assign row_dst_addr_w = {16'h0, dst_off_q} +
                          ({17'h0, row_idx_q} * {21'h0, n_tiles_q}) +
                          {21'h0, write_chunk_q};
  assign ln_param_addr_w = {16'h0, src2_off_q} + {19'h0, read_idx_q};
  assign gelu_i8_addr_w = {16'h0, src1_off_q} +
                          ({17'h0, row_idx_q} * {21'h0, n_tiles_q}) +
                          {21'h0, write_chunk_q};
  assign gelu_acc_addr_w = {16'h0, src1_off_q} +
                           ({17'h0, row_idx_q} * {19'h0, n_chunks_i32_q}) +
                           ({21'h0, write_chunk_q} << 2) +
                           {30'h0, gelu_part_q};
  assign gelu_dst_addr_w = {16'h0, dst_off_q} +
                           ({17'h0, row_idx_q} * {21'h0, n_tiles_q}) +
                           {21'h0, write_chunk_q};
  assign attn_qkt_addr_w = {16'h0, src1_off_q} +
                           ({17'h0, row_idx_q} * {19'h0, k_chunks_i32_q}) +
                           {19'h0, read_idx_q};
  assign attn_v_addr_w = {16'h0, src2_off_q} +
                         ({16'h0, attn_k_idx_q} * {21'h0, n_tiles_q}) +
                         {19'h0, read_idx_q};

  always_comb begin
    row_write_data_w = 128'h0;
    gelu_i8_write_data_w = 128'h0;
    gelu_i32_write_data_w = 128'h0;
    attn_write_data_w = 128'h0;

    for (int lane = 0; lane < 16; lane++) begin
      int    idx;
      fp32_t x_b;
      idx = integer'(write_chunk_q) * 16 + lane;
      if (idx < integer'(n_elems_q))
        row_write_data_w[(lane * 8) +: 8] = out_bytes_q[idx];

      x_b = fp32_mul_bits(fp32_from_i8(get_i8(gelu_i8_row_q, lane)), scale0_q);
      gelu_i8_write_data_w[(lane * 8) +: 8] = quantize_to_i8(fp32_gelu_bits(x_b), scale1_q);

      if (lane < 4) begin
        x_b = fp32_mul_bits(fp32_from_i32(get_i32(gelu_row0_q, lane)), scale0_q);
        gelu_i32_write_data_w[(lane * 8) +: 8] = quantize_to_i8(fp32_gelu_bits(x_b), scale1_q);
        x_b = fp32_mul_bits(fp32_from_i32(get_i32(gelu_row1_q, lane)), scale0_q);
        gelu_i32_write_data_w[((lane + 4) * 8) +: 8] = quantize_to_i8(fp32_gelu_bits(x_b), scale1_q);
        x_b = fp32_mul_bits(fp32_from_i32(get_i32(gelu_row2_q, lane)), scale0_q);
        gelu_i32_write_data_w[((lane + 8) * 8) +: 8] = quantize_to_i8(fp32_gelu_bits(x_b), scale1_q);
        x_b = fp32_mul_bits(fp32_from_i32(get_i32(gelu_row3_q, lane)), scale0_q);
        gelu_i32_write_data_w[((lane + 12) * 8) +: 8] = quantize_to_i8(fp32_gelu_bits(x_b), scale1_q);
      end

      idx = integer'(write_chunk_q) * 16 + lane;
      if (idx < integer'(n_elems_q))
        attn_write_data_w[(lane * 8) +: 8] = quantize_to_i8(attn_accum_q[idx], scale2_q);
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
      n_chunks_i32_q <= 13'h0;
      k_chunks_i32_q <= 13'h0;
      n_elems_q      <= 16'h0;
      k_elems_q      <= 16'h0;
      ln_gamma_rows_q<= 16'h0;
      ln_param_rows_q<= 16'h0;
      fault_code_r   <= 4'(FAULT_NONE);
      row_idx_q      <= 15'h0;
      read_idx_q     <= 13'h0;
      write_chunk_q  <= 11'h0;
      gelu_part_q    <= 2'h0;
      attn_k_idx_q   <= 16'h0;
      gelu_i8_row_q  <= 128'h0;
      gelu_row0_q    <= 128'h0;
      gelu_row1_q    <= 128'h0;
      gelu_row2_q    <= 128'h0;
      gelu_row3_q    <= 128'h0;
      row_write_q    <= 128'h0;
      scale0_q       <= 32'd0;
      scale1_q       <= 32'd0;
      scale2_q       <= 32'd0;
      scale3_q       <= 32'd0;
      attn_row_max_q <= 32'd0;
      attn_exp_sum_q <= 32'd0;
      ln_debug_mean_q <= 32'd0;
      ln_debug_var_q <= 32'd0;
      ln_debug_denom_q <= 32'd0;
      for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
        row_data_q[i] <= 32'd0;
        attn_accum_q[i] <= 32'd0;
        gamma_q[i]    <= 32'd0;
        beta_q[i]     <= 32'd0;
        out_bytes_q[i] <= 8'h00;
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
            n_chunks_i32_q  <= dispatch_n_chunks_i32_w;
            k_chunks_i32_q  <= dispatch_k_chunks_i32_w;
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
            write_chunk_q   <= 11'h0;
            gelu_part_q     <= 2'h0;
            attn_k_idx_q    <= 16'h0;

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
                    state <= F_ROW_I32_REQ;
                  else
                    state <= F_ROW_I8_REQ;
                end

                OP_LAYERNORM:
                  state <= F_LN_PARAM_REQ;

                OP_GELU: begin
                  if (src1_buf == BUF_ACCUM)
                    state <= F_GELU_I32_REQ;
                  else
                    state <= F_GELU_I8_REQ;
                end

                OP_SOFTMAX_ATTNV:
                  state <= F_ATTN_QKT_REQ;

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
            state      <= F_ROW_I8_REQ;
          end
        end

        F_ROW_I8_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else begin
            state <= F_ROW_I8_LATCH;
          end
        end

        F_ROW_I8_LATCH: begin
          integer base_idx;
          base_idx = integer'(read_idx_q) * 16;
          for (int lane = 0; lane < 16; lane++) begin
            if ((base_idx + lane) < integer'(n_elems_q))
              row_data_q[base_idx + lane] <=
                  fp32_mul_bits(fp32_from_i8(get_i8(sram_b_rdata, lane)), scale0_q);
          end

          if (read_idx_q + 13'd1 < {2'h0, n_tiles_q}) begin
            read_idx_q <= read_idx_q + 13'd1;
            state      <= F_ROW_I8_REQ;
          end else begin
            write_chunk_q <= 11'h0;
            state         <= F_ROW_COMPUTE;
          end
        end

        F_ROW_I32_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else begin
            state <= F_ROW_I32_LATCH;
          end
        end

        F_ROW_I32_LATCH: begin
          integer base_idx;
          base_idx = integer'(read_idx_q) * 4;
          for (int lane = 0; lane < 4; lane++) begin
            if ((base_idx + lane) < integer'(n_elems_q))
              row_data_q[base_idx + lane] <=
                  fp32_mul_bits(fp32_from_i32(get_i32(sram_b_rdata, lane)), scale0_q);
          end

          if (read_idx_q + 13'd1 < n_chunks_i32_q) begin
            read_idx_q <= read_idx_q + 13'd1;
            state      <= F_ROW_I32_REQ;
          end else begin
            write_chunk_q <= 11'h0;
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
                out_bytes_q[i] <= quantize_to_i8(fp32_div_bits(exp_b, exp_sum_b), scale1_q);
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
                out_bytes_q[i] <= quantize_to_i8(y_b, scale1_q);
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
          end else if (write_chunk_q + 11'd1 < n_tiles_q) begin
            write_chunk_q <= write_chunk_q + 11'd1;
            state         <= F_ROW_PACK;
          end else if (row_idx_q + 15'd1 < m_rows_q) begin
            row_idx_q     <= row_idx_q + 15'd1;
            read_idx_q    <= 13'h0;
            write_chunk_q <= 11'h0;
            if ((opcode_q == OP_SOFTMAX) && (src1_buf_q == BUF_ACCUM))
              state <= F_ROW_I32_REQ;
            else
              state <= F_ROW_I8_REQ;
          end else begin
            state <= F_IDLE;
          end
        end

        F_GELU_I8_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else begin
            state <= F_GELU_I8_LATCH;
          end
        end

        F_GELU_I8_LATCH: begin
          gelu_i8_row_q <= sram_b_rdata;
          state         <= F_GELU_I8_WRITE;
        end

        F_GELU_I8_WRITE: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else if (write_chunk_q + 11'd1 < n_tiles_q) begin
            write_chunk_q <= write_chunk_q + 11'd1;
            state         <= F_GELU_I8_REQ;
          end else if (row_idx_q + 15'd1 < m_rows_q) begin
            row_idx_q     <= row_idx_q + 15'd1;
            write_chunk_q <= 11'h0;
            state         <= F_GELU_I8_REQ;
          end else begin
            state <= F_IDLE;
          end
        end

        F_GELU_I32_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else begin
            state <= F_GELU_I32_LATCH;
          end
        end

        F_GELU_I32_LATCH: begin
          case (gelu_part_q)
            2'd0: gelu_row0_q <= sram_b_rdata;
            2'd1: gelu_row1_q <= sram_b_rdata;
            2'd2: gelu_row2_q <= sram_b_rdata;
            default: gelu_row3_q <= sram_b_rdata;
          endcase

          if (gelu_part_q == 2'd3) begin
            state <= F_GELU_I32_WRITE;
          end else begin
            gelu_part_q <= gelu_part_q + 2'd1;
            state       <= F_GELU_I32_REQ;
          end
        end

        F_GELU_I32_WRITE: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else if (write_chunk_q + 11'd1 < n_tiles_q) begin
            write_chunk_q <= write_chunk_q + 11'd1;
            gelu_part_q   <= 2'h0;
            state         <= F_GELU_I32_REQ;
          end else if (row_idx_q + 15'd1 < m_rows_q) begin
            row_idx_q     <= row_idx_q + 15'd1;
            write_chunk_q <= 11'h0;
            gelu_part_q   <= 2'h0;
            state         <= F_GELU_I32_REQ;
          end else begin
            state <= F_IDLE;
          end
        end

        F_ATTN_QKT_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else begin
            state <= F_ATTN_QKT_LATCH;
          end
        end

        F_ATTN_QKT_LATCH: begin
          integer base_idx;
          base_idx = integer'(read_idx_q) * 4;
          for (int lane = 0; lane < 4; lane++) begin
            if ((base_idx + lane) < integer'(k_elems_q))
              row_data_q[base_idx + lane] <=
                  fp32_mul_bits(fp32_from_i32(get_i32(sram_b_rdata, lane)), scale0_q);
          end

          if (read_idx_q + 13'd1 < k_chunks_i32_q) begin
            read_idx_q <= read_idx_q + 13'd1;
            state      <= F_ATTN_QKT_REQ;
          end else begin
            state <= F_ATTN_PREP;
          end
        end

        F_ATTN_PREP: begin
          fp32_t row_max_b;
          fp32_t exp_sum_b;
          row_max_b = row_data_q[0];
          for (int i = 1; i < SFU_MAX_ROW_ELEMS; i++) begin
            if ((i < integer'(k_elems_q)) && fp32_gt(row_data_q[i], row_max_b))
              row_max_b = row_data_q[i];
          end

          exp_sum_b = 32'd0;
          for (int i = 0; i < SFU_MAX_ROW_ELEMS; i++) begin
            if (i < integer'(k_elems_q))
              exp_sum_b = fp32_add_bits(
                  exp_sum_b, fp32_exp_bits(fp32_sub_bits(row_data_q[i], row_max_b)));
            if (i < integer'(n_elems_q))
              attn_accum_q[i] <= 32'd0;
          end

          attn_row_max_q <= row_max_b;
          attn_exp_sum_q <= exp_sum_b;
          attn_k_idx_q   <= 16'h0;
          read_idx_q     <= 13'h0;
          write_chunk_q  <= 11'h0;
          state          <= F_ATTN_V_REQ;
        end

        F_ATTN_V_REQ: begin
          if (sram_b_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else begin
            state <= F_ATTN_V_LATCH;
          end
        end

        F_ATTN_V_LATCH: begin
          fp32_t weight_b;
          weight_b = fp32_div_bits(
              fp32_exp_bits(fp32_sub_bits(row_data_q[integer'(attn_k_idx_q)], attn_row_max_q)),
              attn_exp_sum_q);
          for (int lane = 0; lane < 16; lane++) begin
            integer idx;
            idx = integer'(read_idx_q) * 16 + lane;
            if (idx < integer'(n_elems_q))
              attn_accum_q[idx] <= fp32_add_bits(
                  attn_accum_q[idx],
                  fp32_mul_bits(
                      fp32_mul_bits(weight_b, fp32_from_i8(get_i8(sram_b_rdata, lane))),
                      scale1_q));
          end

          if (read_idx_q + 13'd1 < {2'h0, n_tiles_q}) begin
            read_idx_q <= read_idx_q + 13'd1;
            state      <= F_ATTN_V_REQ;
          end else if (attn_k_idx_q + 16'd1 < k_elems_q) begin
            attn_k_idx_q <= attn_k_idx_q + 16'd1;
            read_idx_q   <= 13'h0;
            state        <= F_ATTN_V_REQ;
          end else begin
            write_chunk_q <= 11'h0;
            state         <= F_ATTN_WRITE;
          end
        end

        F_ATTN_WRITE: begin
          if (sram_a_fault) begin
            fault_code_r <= 4'(FAULT_SRAM_OOB);
            state        <= F_FAULT;
          end else if (write_chunk_q + 11'd1 < n_tiles_q) begin
            write_chunk_q <= write_chunk_q + 11'd1;
            state         <= F_ATTN_WRITE;
          end else if (row_idx_q + 15'd1 < m_rows_q) begin
            row_idx_q     <= row_idx_q + 15'd1;
            read_idx_q    <= 13'h0;
            write_chunk_q <= 11'h0;
            state         <= F_ATTN_QKT_REQ;
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

      F_ROW_I8_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src1_buf_q;
        sram_b_row = row_i8_addr_w[15:0];
      end

      F_ROW_I32_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src1_buf_q;
        sram_b_row = row_i32_addr_w[15:0];
      end

      F_ROW_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = row_dst_addr_w[15:0];
        sram_a_wdata = row_write_q;
      end

      F_GELU_I8_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src1_buf_q;
        sram_b_row = gelu_i8_addr_w[15:0];
      end

      F_GELU_I8_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = gelu_dst_addr_w[15:0];
        sram_a_wdata = gelu_i8_write_data_w;
      end

      F_GELU_I32_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src1_buf_q;
        sram_b_row = gelu_acc_addr_w[15:0];
      end

      F_GELU_I32_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = gelu_dst_addr_w[15:0];
        sram_a_wdata = gelu_i32_write_data_w;
      end

      F_ATTN_QKT_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src1_buf_q;
        sram_b_row = attn_qkt_addr_w[15:0];
      end

      F_ATTN_V_REQ: begin
        sram_b_en  = 1'b1;
        sram_b_buf = src2_buf_q;
        sram_b_row = attn_v_addr_w[15:0];
      end

      F_ATTN_WRITE: begin
        sram_a_en    = 1'b1;
        sram_a_we    = 1'b1;
        sram_a_buf   = dst_buf_q;
        sram_a_row   = row_dst_addr_w[15:0];
        sram_a_wdata = attn_write_data_w;
      end

      default: ;
    endcase
  end

endmodule

`endif // SFU_ENGINE_SV
