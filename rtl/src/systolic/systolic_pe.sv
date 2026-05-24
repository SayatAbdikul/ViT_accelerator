`ifndef SYSTOLIC_PE_SV
`define SYSTOLIC_PE_SV

`include "fp32_prim_pkg.sv"

// One processing element in the 16x16 systolic mesh (W8A16 datapath).
//
// FP16 inputs are widened to FP32, multiplied (RNE FP32), then added to a
// 32-bit FP32 accumulator (RNE FP32). The MAC is NOT a fused FMA -- mul and
// add round independently. This matches the golden simulator's per-K-step
// reduction in software/taccel/golden_model/systolic_w8a16.py and the
// software-side reference in software/taccel/utils/fp32_prim_ref.py.
//
// Forwarding (a_out / b_out) is plain register pass-through so the chained
// mode skew pipeline keeps working. The widen+mul+add chain is purely
// combinational; if synthesis timing requires it, pipeline later -- the
// bit-exact contract is unaffected.

module systolic_pe
  import fp32_prim_pkg::*;
(
  input  logic         clk,
  input  logic         rst_n,
  input  logic         en,
  input  logic         acc_clear,
  input  logic [15:0]  a_in,
  input  logic [15:0]  b_in,
  output logic [15:0]  a_out,
  output logic [15:0]  b_out,
  output logic [31:0]  acc
);

  fp32_t a_fp32, b_fp32, prod_fp32, acc_next;

  always_comb begin
    a_fp32    = fp32_from_fp16_bits(a_in);
    b_fp32    = fp32_from_fp16_bits(b_in);
    prod_fp32 = fp32_mul_bits(a_fp32, b_fp32);
    acc_next  = fp32_add_bits(acc, prod_fp32);
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a_out <= 16'h0;
      b_out <= 16'h0;
      acc   <= 32'h0;
    end else begin
      if (acc_clear) begin
        a_out <= 16'h0;
        b_out <= 16'h0;
        acc   <= 32'h0;
      end else if (en) begin
        a_out <= a_in;
        b_out <= b_in;
        acc   <= acc_next;
      end
    end
  end

endmodule

`endif // SYSTOLIC_PE_SV
