`ifndef SYSTOLIC_ARRAY_SV
`define SYSTOLIC_ARRAY_SV

`include "taccel_pkg.sv"

// 16x16 systolic mesh wrapper (W8A16 datapath).
//
// A 16-lane row is 16 FP16 = 32 bytes = 256 bits, twice the SRAM row width.
// The controller assembles two 128-bit SRAM reads into the 256-bit row vector
// presented here, then pulses step_en once per K-strip; the array is unaware
// of the assembly cadence. Internally each lane is unpacked into FP16 and
// passed to a PE that widens to FP32 before MAC. ACC is FP32 (32 bits).
//
// Modes:
//   - broadcast: each row/column sees the same lane value
//   - chained:   operands flow across the mesh one PE per cycle, with skew
//                registers (now 16-bit FP16) on the boundary

module systolic_array
  import taccel_pkg::*;
#(
  parameter int SYSTOLIC_ARCH_MODE = SYS_MODE_DEFAULT
)
(
  input  logic                          clk,
  input  logic                          rst_n,
  input  logic                          step_en,
  input  logic                          clear_acc,
  input  logic [SYS_DIM*16-1:0]         a_row_data,   // 16 FP16 lanes
  input  logic [SYS_DIM*16-1:0]         b_row_data,   // 16 FP16 lanes
  output logic [SYS_DIM*SYS_DIM*32-1:0] acc_flat      // 16x16 FP32 accs
);

  logic [15:0] a_vec [0:SYS_DIM-1];
  logic [15:0] b_vec [0:SYS_DIM-1];
  logic [15:0] a_edge_vec [0:SYS_DIM-1];
  logic [15:0] b_edge_vec [0:SYS_DIM-1];
  logic [15:0] a_skew [0:SYS_DIM-1][0:SYS_DIM-2];
  logic [15:0] b_skew [0:SYS_DIM-1][0:SYS_DIM-2];

  // PE-local state and interconnect signals.
  logic [31:0] pe_acc   [0:SYS_DIM-1][0:SYS_DIM-1];
  logic [15:0] pe_a_in  [0:SYS_DIM-1][0:SYS_DIM-1];
  logic [15:0] pe_b_in  [0:SYS_DIM-1][0:SYS_DIM-1];
  logic [15:0] pe_a_out [0:SYS_DIM-1][0:SYS_DIM-1];
  logic [15:0] pe_b_out [0:SYS_DIM-1][0:SYS_DIM-1];

  genvar i, j;
  // Unpack the incoming 256-bit rows into 16 FP16 lanes and select the
  // edge-fed values used in chained mode after skew insertion.
  generate
    for (i = 0; i < SYS_DIM; i++) begin : GEN_A_B
      assign a_vec[i] = a_row_data[i*16 +: 16];
      assign b_vec[i] = b_row_data[i*16 +: 16];

      if (i == 0) begin : GEN_EDGE_NO_DELAY
        assign a_edge_vec[i] = a_vec[i];
        assign b_edge_vec[i] = b_vec[i];
      end else begin : GEN_EDGE_DELAYED
        assign a_edge_vec[i] = a_skew[i][i-1];
        assign b_edge_vec[i] = b_skew[i][i-1];
      end
    end
  endgenerate

  // Chained systolic mode requires boundary skew so A/B operands that belong
  // to the same k arrive at each PE on the same cycle. Skew regs are FP16 +0
  // on reset/clear (16'h0 = FP16 +0.0, which is also FP32 +0.0 after widen).
  always_ff @(posedge clk or negedge rst_n) begin : SKew_PIPE
    int r, s;
    if (!rst_n) begin
      for (r = 0; r < SYS_DIM; r++) begin
        for (s = 0; s < SYS_DIM-1; s++) begin
          a_skew[r][s] <= 16'h0;
          b_skew[r][s] <= 16'h0;
        end
      end
    end else if (clear_acc) begin
      for (r = 0; r < SYS_DIM; r++) begin
        for (s = 0; s < SYS_DIM-1; s++) begin
          a_skew[r][s] <= 16'h0;
          b_skew[r][s] <= 16'h0;
        end
      end
    end else if (step_en) begin
      for (r = 0; r < SYS_DIM; r++) begin
        a_skew[r][0] <= a_vec[r];
        b_skew[r][0] <= b_vec[r];
        for (s = 1; s < SYS_DIM-1; s++) begin
          a_skew[r][s] <= a_skew[r][s-1];
          b_skew[r][s] <= b_skew[r][s-1];
        end
      end
    end
  end

  // Dual-mode routing scaffold:
  // - Broadcast mode: all PEs in row/col see same a_vec/b_vec lane.
  // - Chained mode: left/top edge injects a_vec/b_vec and interior PEs consume
  //   neighbor outputs (west/east for A, north/south for B).
  generate
    for (i = 0; i < SYS_DIM; i++) begin : GEN_ROUTE_ROW
      for (j = 0; j < SYS_DIM; j++) begin : GEN_ROUTE_COL
        always_comb begin
          if (SYSTOLIC_ARCH_MODE == SYS_MODE_CHAINED) begin
            pe_a_in[i][j] = (j == 0) ? a_edge_vec[i] : pe_a_out[i][j-1];
            pe_b_in[i][j] = (i == 0) ? b_edge_vec[j] : pe_b_out[i-1][j];
          end else begin
            pe_a_in[i][j] = a_vec[i];
            pe_b_in[i][j] = b_vec[j];
          end
        end
      end
    end
  endgenerate

  // Instantiate the full mesh and flatten the FP32 accumulator matrix for
  // the controller's writeback logic.
  generate
    for (i = 0; i < SYS_DIM; i++) begin : GEN_ROW
      for (j = 0; j < SYS_DIM; j++) begin : GEN_COL
        systolic_pe u_pe (
          .clk       (clk),
          .rst_n     (rst_n),
          .en        (step_en),
          .acc_clear (clear_acc),
          .a_in      (pe_a_in[i][j]),
          .b_in      (pe_b_in[i][j]),
          .a_out     (pe_a_out[i][j]),
          .b_out     (pe_b_out[i][j]),
          .acc       (pe_acc[i][j])
        );

        localparam int FLAT = (i * SYS_DIM + j) * 32;
        assign acc_flat[FLAT +: 32] = pe_acc[i][j];
      end
    end
  endgenerate

endmodule

`endif // SYSTOLIC_ARRAY_SV
