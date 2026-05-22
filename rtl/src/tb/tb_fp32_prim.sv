`ifndef TB_FP32_PRIM_SV
`define TB_FP32_PRIM_SV

module tb_fp32_prim
  import fp32_prim_pkg::*;
(
  input  logic [3:0]  op,
  input  logic [31:0] a_bits,
  input  logic [31:0] b_bits,
  output logic [31:0] result_bits,
  output logic signed [31:0] q_i32   // INT8 quantize result (op=10)
);

  always_comb begin
    q_i32 = 32'sd0;
    result_bits = FP32_QNAN_BITS;
    unique case (op)
      4'd0:  result_bits = fp32_round_bits(a_bits);
      4'd1:  result_bits = fp32_add_bits(a_bits, b_bits);
      4'd2:  result_bits = fp32_sub_bits(a_bits, b_bits);
      4'd3:  result_bits = fp32_mul_bits(a_bits, b_bits);
      4'd4:  result_bits = fp32_div_bits(a_bits, b_bits);
      4'd5:  result_bits = fp32_sqrt_bits(a_bits);
      4'd6:  result_bits = fp32_exp_bits(a_bits);
      4'd7:  result_bits = fp32_erf_bits(a_bits);
      4'd8:  result_bits = fp32_gelu_bits(a_bits);
      4'd9:  result_bits = fp32_from_fp16_bits(a_bits[15:0]);
      4'd10: begin
               result_bits = 32'd0;
               q_i32 = 32'(fp32_quantize_i8_bits(a_bits, b_bits));
             end
      default: result_bits = FP32_QNAN_BITS;
    endcase
  end

endmodule

`endif
