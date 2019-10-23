/*
* bsg_axil_rxs.v
*             ___     ___     ___     ___     ___
* clk     ___/   \___/   \___/   \___/   \___/   \___
*             _______
* araddr  XXXX_ADDR__XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
*             _______
* arvalid ___/       \_______________________________
*         ___________________________________________
* arready
*                                     _______
* rdata   XXXXXXXXXXXXXXXXXXXXXXXXXXXX_DATA__XXXXXXXX
*                                     _______
* rresp   XXXXXXXXXXXXXXXXXXXXXXXXXXXX_RESP__XXXXXXXX
*                                     _______
* rvalid  ___________________________/       \_______
*         ___________________________________________
* rready
*
*/

`include "bsg_defines.v"
`include "bsg_manycore_link_to_axil_pkg.v"

module bsg_axil_rxs
  import bsg_manycore_link_to_axil_pkg::*;
#(parameter num_fifos_p = "inv") (
  input                          clk_i
  ,input                          reset_i
  // axil tx channel
  ,input  [           31:0]       araddr_i
  ,input                          arvalid_i
  ,output                         arready_o
  ,output [           31:0]       rdata_o
  ,output [            1:0]       rresp_o
  ,output                         rvalid_o
  ,input                          rready_i
  // fifo
  ,input  [num_fifos_p-1:0][31:0] rxs_i
  // ,input  [num_fifos_p-1:0]       rxs_v_i
  // from fifo, ready only, data will be zeros if not valid
  ,output [num_fifos_p-1:0]       rxs_ready_o
  // read monitor registers and rom
  ,output [           31:0]       rd_addr_o
  ,input  [num_fifos_p-1:0][31:0] mm2s_regs_i
  ,input  [           31:0]       mcl_data_i
);


  // tie unused signal
  // wire [num_fifos_p-1:0] unused_fifo_valid_li = rxs_v_i;

  // --------------------------------------------
  // axil read state machine
  // --------------------------------------------

  typedef enum bit [1:0] {
    E_RD_IDLE = 2'b0,
    E_RD_ADDR = 2'd1,
    E_RD_DATA = 2'd2
  } rd_state_e;

  rd_state_e rd_state_r, rd_state_n;

  always_comb begin
    rd_state_n = rd_state_r;
    case (rd_state_r)

      E_RD_IDLE : begin
        if (arvalid_i)
          rd_state_n = E_RD_ADDR;
      end

      E_RD_ADDR : begin
        rd_state_n = E_RD_DATA;  // always ready to accept address
      end

      E_RD_DATA : begin
        if (rvalid_o & rready_i)
          rd_state_n = E_RD_IDLE;
      end

      default : rd_state_n = E_RD_IDLE;
    endcase
  end

  always_ff @(posedge clk_i) begin
    if (reset_i)
      rd_state_r <= E_RD_IDLE;
    else
      rd_state_r <= rd_state_n;
  end


  logic [num_fifos_p-1:0] base_addr_hit;
  logic [num_fifos_p-1:0] fifo_addr_v;
  logic                   rom_addr_hit ;

  wire in_rd_addr = rd_state_r == E_RD_ADDR;
  wire in_rd_data = rd_state_r == E_RD_DATA;

  logic [31:0] rd_addr_r, rd_addr_n;
  logic [ 1:0] rresp_n  ;

  always_comb begin
    // raddr channel
    rd_addr_n  = (arvalid_i & arready_o) ? araddr_i : rd_addr_r;
    // read response, raise DECERR if access address is out of range
    rresp_n    = ~(|base_addr_hit | rom_addr_hit) & in_rd_data ? 2'b11 : '0; // DECERR or OKAY
  end

  always_ff @(posedge clk_i) begin
    if (reset_i) begin
      rd_addr_r <= '0;
    end
    else begin
      rd_addr_r <= rd_addr_n;
    end
  end

  for (genvar i=0; i<num_fifos_p; i++) begin : rd_addr_hit
    // axil rd addr hits the slot base address
    assign base_addr_hit[i] = in_rd_data
      && (rd_addr_r[axil_base_addr_width_gp+:axil_slot_idx_width_gp]
        == axil_slot_idx_width_gp'(i+(axil_m_slot_addr_gp>>axil_base_addr_width_gp)));
    // rd command is read data register
    assign fifo_addr_v[i] = base_addr_hit[i]
      && (rd_addr_r[0+:axil_base_addr_width_gp]
        == axil_base_addr_width_gp'(axil_mm2s_ofs_rdr_gp));
  end : rd_addr_hit

  // axil rd addr hits the monitor base address
  assign rom_addr_hit = in_rd_data
    && (rd_addr_r[axil_base_addr_width_gp+:axil_slot_idx_width_gp]
      == axil_slot_idx_width_gp'(num_fifos_p+(axil_m_slot_addr_gp>>axil_base_addr_width_gp)));

  wire read_fifo = |fifo_addr_v; // get the fifo data when any fifo is ready

  // output signals

  // axil side
  assign arready_o = in_rd_addr;
  assign rvalid_o  = in_rd_data;  //always valid for the read

  assign rresp_o = rresp_n;

  // select the read data
  if (num_fifos_p == 1) begin : one_fifo
    assign rd_fifo_idx = fifo_addr_v;
    assign rdata_o = rom_addr_hit ? mm2s_regs_i
                                : read_fifo ? rxs_i
                                            : mm2s_regs_i;
  end
  else begin : many_fifos
    logic [`BSG_SAFE_CLOG2(num_fifos_p)-1:0] rd_slot_addr_lo;

    bsg_encode_one_hot #(.width_p(num_fifos_p)) axil_slot_idx_encode (
      .i(base_addr_hit)
      ,.addr_o(rd_slot_addr_lo)
      ,.v_o()
    );
    assign rdata_o = rom_addr_hit ? mcl_data_i
                                : read_fifo ? rxs_i[rd_slot_addr_lo]
                                            : mm2s_regs_i[rd_slot_addr_lo];
  end

  // tx stream side
  assign rxs_ready_o = {num_fifos_p{rvalid_o & rready_i}} & fifo_addr_v;

  assign rd_addr_o = fifo_addr_v ? '0 : rd_addr_r;

endmodule