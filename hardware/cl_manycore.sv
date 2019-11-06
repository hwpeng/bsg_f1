// Copyright (c) 2019, University of Washington All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this list
// of conditions and the following disclaimer.
//
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
//
// Neither the name of the copyright holder nor the names of its contributors may
// be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/**
 *  cl_manycore.v
 */

`include "bsg_bladerunner_rom_pkg.vh"

module cl_manycore
  import cl_manycore_pkg::*;
   import bsg_bladerunner_rom_pkg::*;
   import bsg_bladerunner_mem_cfg_pkg::*;
   (
`include "cl_ports.vh"
    );

   // For some silly reason, you need to leave this up here...
   logic rst_main_n_sync;

`include "bsg_defines.v"
`include "bsg_manycore_packet.vh"
`include "cl_id_defines.vh"
`include "cl_manycore_defines.vh"

   //--------------------------------------------
   // Start with Tie-Off of Unused Interfaces
   //---------------------------------------------
   // The developer should use the next set of `include to properly tie-off any
   // unused interface The list is put in the top of the module to avoid cases
   // where developer may forget to remove it from the end of the file

`include "unused_flr_template.inc"
`include "unused_ddr_a_b_d_template.inc"
   //`include "unused_ddr_c_template.inc"
`include "unused_pcim_template.inc"
`include "unused_dma_pcis_template.inc"
`include "unused_cl_sda_template.inc"
`include "unused_sh_bar1_template.inc"
`include "unused_apppf_irq_template.inc"

   localparam lc_clk_main_a0_p = 8000; // 8000 is 125 MHz

   //-------------------------------------------------
   // Wires
   //-------------------------------------------------
   logic pre_sync_rst_n;

   logic [15:0] vled_q;
   logic [15:0] pre_cl_sh_status_vled;
   logic [15:0] sh_cl_status_vdip_q;
   logic [15:0] sh_cl_status_vdip_q2;

   //-------------------------------------------------
   // PCI ID Values
   //-------------------------------------------------
   assign cl_sh_id0[31:0] = `CL_SH_ID0;
   assign cl_sh_id1[31:0] = `CL_SH_ID1;

   //-------------------------------------------------
   // Reset Synchronization
   //-------------------------------------------------

   always_ff @(negedge rst_main_n or posedge clk_main_a0)
     if (!rst_main_n)
       begin
          pre_sync_rst_n  <= 0;
          rst_main_n_sync <= 0;
       end
     else
       begin
          pre_sync_rst_n  <= 1;
          rst_main_n_sync <= pre_sync_rst_n;
       end

   //-------------------------------------------------
   // Virtual LED Register
   //-------------------------------------------------
   // Flop/synchronize interface signals
   always_ff @(posedge clk_main_a0)
     if (!rst_main_n_sync) begin
        sh_cl_status_vdip_q[15:0]  <= 16'h0000;
        sh_cl_status_vdip_q2[15:0] <= 16'h0000;
        cl_sh_status_vled[15:0]    <= 16'h0000;
     end
     else begin
        sh_cl_status_vdip_q[15:0]  <= sh_cl_status_vdip[15:0];
        sh_cl_status_vdip_q2[15:0] <= sh_cl_status_vdip_q[15:0];
        cl_sh_status_vled[15:0]    <= pre_cl_sh_status_vled[15:0];
     end

   // The register contains 16 read-only bits corresponding to 16 LED's.
   // The same LED values can be read from the CL to Shell interface
   // by using the linux FPGA tool: $ fpga-get-virtual-led -S 0
   always_ff @(posedge clk_main_a0)
     if (!rst_main_n_sync) begin
        vled_q[15:0] <= 16'h0000;
     end
     else begin
        vled_q[15:0] <= 16'hbeef;
     end

   assign pre_cl_sh_status_vled[15:0] = vled_q[15:0];
   assign cl_sh_status0[31:0] = 32'h0;
   assign cl_sh_status1[31:0] = 32'h0;

   //-------------------------------------------------
   // Post-Pipeline-Register OCL AXI-L Signals
   //-------------------------------------------------
   logic        m_axil_ocl_awvalid;
   logic [31:0] m_axil_ocl_awaddr;
   logic        m_axil_ocl_awready;

   logic        m_axil_ocl_wvalid;
   logic [31:0] m_axil_ocl_wdata;
   logic [ 3:0] m_axil_ocl_wstrb;
   logic        m_axil_ocl_wready;

   logic        m_axil_ocl_bvalid;
   logic [ 1:0] m_axil_ocl_bresp;
   logic        m_axil_ocl_bready;

   logic        m_axil_ocl_arvalid;
   logic [31:0] m_axil_ocl_araddr;
   logic        m_axil_ocl_arready;

   logic        m_axil_ocl_rvalid;
   logic [31:0] m_axil_ocl_rdata;
   logic [ 1:0] m_axil_ocl_rresp;
   logic        m_axil_ocl_rready;

   //--------------------------------------------
   // AXI4 signals for the Manycore
   //---------------------------------------------
   logic [5:0]  m_axi4_manycore_awid;
   logic [63:0] m_axi4_manycore_awaddr;
   logic [7:0]  m_axi4_manycore_awlen;
   logic [2:0]  m_axi4_manycore_awsize;
   logic [1:0]  m_axi4_manycore_awburst;
   logic [0:0]  m_axi4_manycore_awlock;
   logic [3:0]  m_axi4_manycore_awcache;
   logic [2:0]  m_axi4_manycore_awprot;
   logic [3:0]  m_axi4_manycore_awregion;
   logic [3:0]  m_axi4_manycore_awqos;
   logic        m_axi4_manycore_awvalid;
   logic        m_axi4_manycore_awready;

   logic [511:0] m_axi4_manycore_wdata;
   logic [63:0]  m_axi4_manycore_wstrb;
   logic         m_axi4_manycore_wlast;
   logic         m_axi4_manycore_wvalid;
   logic         m_axi4_manycore_wready;

   logic [5:0]   m_axi4_manycore_bid;
   logic [1:0]   m_axi4_manycore_bresp;
   logic         m_axi4_manycore_bvalid;
   logic         m_axi4_manycore_bready;

   logic [5:0]   m_axi4_manycore_arid;
   logic [63:0]  m_axi4_manycore_araddr;
   logic [7:0]   m_axi4_manycore_arlen;
   logic [2:0]   m_axi4_manycore_arsize;
   logic [1:0]   m_axi4_manycore_arburst;
   logic [0:0]   m_axi4_manycore_arlock;
   logic [3:0]   m_axi4_manycore_arcache;
   logic [2:0]   m_axi4_manycore_arprot;
   logic [3:0]   m_axi4_manycore_arregion;
   logic [3:0]   m_axi4_manycore_arqos;
   logic         m_axi4_manycore_arvalid;
   logic         m_axi4_manycore_arready;

   logic [5:0]   m_axi4_manycore_rid;
   logic [511:0] m_axi4_manycore_rdata;
   logic [1:0]   m_axi4_manycore_rresp;
   logic         m_axi4_manycore_rlast;
   logic         m_axi4_manycore_rvalid;
   logic         m_axi4_manycore_rready;


   //--------------------------------------------
   // AXI-Lite OCL System
   //---------------------------------------------
   axi_register_slice_light
     AXIL_OCL_REG_SLC
       (
        .aclk          (clk_main_a0),
        .aresetn       (rst_main_n_sync),
        .s_axi_awaddr  (sh_ocl_awaddr),
        .s_axi_awprot  (3'h0),
        .s_axi_awvalid (sh_ocl_awvalid),
        .s_axi_awready (ocl_sh_awready),
        .s_axi_wdata   (sh_ocl_wdata),
        .s_axi_wstrb   (sh_ocl_wstrb),
        .s_axi_wvalid  (sh_ocl_wvalid),
        .s_axi_wready  (ocl_sh_wready),
        .s_axi_bresp   (ocl_sh_bresp),
        .s_axi_bvalid  (ocl_sh_bvalid),
        .s_axi_bready  (sh_ocl_bready),
        .s_axi_araddr  (sh_ocl_araddr),
        .s_axi_arvalid (sh_ocl_arvalid),
        .s_axi_arready (ocl_sh_arready),
        .s_axi_rdata   (ocl_sh_rdata),
        .s_axi_rresp   (ocl_sh_rresp),
        .s_axi_rvalid  (ocl_sh_rvalid),
        .s_axi_rready  (sh_ocl_rready),
        .m_axi_awaddr  (m_axil_ocl_awaddr),
        .m_axi_awprot  (),
        .m_axi_awvalid (m_axil_ocl_awvalid),
        .m_axi_awready (m_axil_ocl_awready),
        .m_axi_wdata   (m_axil_ocl_wdata),
        .m_axi_wstrb   (m_axil_ocl_wstrb),
        .m_axi_wvalid  (m_axil_ocl_wvalid),
        .m_axi_wready  (m_axil_ocl_wready),
        .m_axi_bresp   (m_axil_ocl_bresp),
        .m_axi_bvalid  (m_axil_ocl_bvalid),
        .m_axi_bready  (m_axil_ocl_bready),
        .m_axi_araddr  (m_axil_ocl_araddr),
        .m_axi_arvalid (m_axil_ocl_arvalid),
        .m_axi_arready (m_axil_ocl_arready),
        .m_axi_rdata   (m_axil_ocl_rdata),
        .m_axi_rresp   (m_axil_ocl_rresp),
        .m_axi_rvalid  (m_axil_ocl_rvalid),
        .m_axi_rready  (m_axil_ocl_rready)
        );

  `include "bsg_axi_bus_pkg.vh"

    `declare_bsg_axil_bus_s(1, bsg_axil_mosi_bus_s, bsg_axil_miso_bus_s);
    bsg_axil_mosi_bus_s s_axil_mc_li;
    bsg_axil_miso_bus_s s_axil_mc_lo;

    assign s_axil_mc_li.awaddr  = m_axil_ocl_awaddr;
    assign s_axil_mc_li.awvalid = m_axil_ocl_awvalid;
    assign m_axil_ocl_awready   = s_axil_mc_lo.awready;
    assign s_axil_mc_li.wdata   = m_axil_ocl_wdata;
    assign s_axil_mc_li.wstrb   = m_axil_ocl_wstrb;
    assign s_axil_mc_li.wvalid  = m_axil_ocl_wvalid;
    assign m_axil_ocl_wready    = s_axil_mc_lo.wready;
    assign m_axil_ocl_bresp     = s_axil_mc_lo.bresp ;
    assign m_axil_ocl_bvalid    = s_axil_mc_lo.bvalid;
    assign s_axil_mc_li.bready  = m_axil_ocl_bready;
    assign s_axil_mc_li.araddr  = m_axil_ocl_araddr;
    assign s_axil_mc_li.arvalid = m_axil_ocl_arvalid;
    assign m_axil_ocl_arready   = s_axil_mc_lo.arready;
    assign m_axil_ocl_rdata     = s_axil_mc_lo.rdata ;
    assign m_axil_ocl_rresp     = s_axil_mc_lo.rresp ;
    assign m_axil_ocl_rvalid    = s_axil_mc_lo.rvalid;
    assign s_axil_mc_li.rready  = m_axil_ocl_rready;


`ifdef COSIM

   logic         ns_core_clk;
   parameter lc_core_clk_period_p =400000;

   bsg_nonsynth_clock_gen
     #(
       .cycle_time_p(lc_core_clk_period_p)
       )
   core_clk_gen
     (
      .o(ns_core_clk)
      );

`endif


   logic         core_clk;
   logic         core_reset;

`ifdef COSIM
   // This clock mux switches between the "fast" IO Clock and the Slow
   // Unsynthesizable "Core Clk". The assign logic below introduces
   // order-of-evaluation issues that can cause spurrious negedges
   // because the simulator doesn't know what order to evaluate clocks
   // in during a clock switch. See the following datasheet for more
   // information:
   // www.xilinx.com/support/documentation/sw_manuals/xilinx2019_1/ug974-vivado-ultrascale-libraries.pdf
   BUFGMUX
     #(
       .CLK_SEL_TYPE("ASYNC") // SYNC, ASYNC
       )
   BUFGMUX_inst
     (
      .O(core_clk), // 1-bit output: Clock output
      .I0(clk_main_a0), // 1-bit input: Clock input (S=0)
      .I1(ns_core_clk), // 1-bit input: Clock input (S=1)
      .S(sh_cl_status_vdip_q2[0]) // 1-bit input: Clock select
      );

   // THIS IS AN UNSAFE CLOCK CROSSING. It is only guaranteed to work
   // because 1. We're in cosimulation, and 2. we don't have ongoing
   // transfers at the start or end of simulation. This means that
   // core_clk, and clk_main_a0 *are the same signal* (See BUFGMUX
   // above).
   assign core_reset = ~rst_main_n_sync;
`else
   assign core_clk = clk_main_a0;
   assign core_reset = ~rst_main_n_sync;
`endif


   // bladerunner wrapper
   localparam num_axi_slot_lp = (mem_cfg_p == e_vcache_blocking_axi4_xbar_dram ||
                                 mem_cfg_p == e_vcache_blocking_axi4_xbar_model ||
                                 mem_cfg_p == e_vcache_blocking_axi4_hbm) ?
                                num_tiles_x_p : 1;

  `declare_bsg_axi4_bus_s(1, axi_id_width_p, axi_addr_width_p, axi_data_width_p,
                          bsg_axi4_mosi_bus_s, bsg_axi4_miso_bus_s);

    bsg_axi4_mosi_bus_s [num_axi_slot_lp-1:0] axi4_mc_cols_lo;
    bsg_axi4_miso_bus_s [num_axi_slot_lp-1:0] axi4_mc_cols_li;

    // hb_manycore
    bsg_bladerunner_wrapper #(.num_axi_slot_p(num_axi_slot_lp)) hb_mc_wrapper (
      .clk_i       (core_clk        ),
      .reset_i     (core_reset      ),
      .clk2_i      (clk_main_a0     ),
      .reset2_i    (~rst_main_n_sync),
      .s_axil_bus_i(s_axil_mc_li    ),
      .s_axil_bus_o(s_axil_mc_lo    ),
      .m_axi4_bus_o(axi4_mc_cols_lo ),
      .m_axi4_bus_i(axi4_mc_cols_li )
    );


  // LEVEL 3
  //
  if (mem_cfg_p == e_vcache_blocking_axi4_f1_dram ||
      mem_cfg_p == e_vcache_blocking_axi4_f1_model) begin : lv3_axi4_c
   // Attach cache to output DRAM

    // AXI Address Write signals
    assign m_axi4_manycore_awid     = axi4_mc_cols_lo[0].awid;
    assign m_axi4_manycore_awaddr   = axi4_mc_cols_lo[0].awaddr;
    assign m_axi4_manycore_awvalid  = axi4_mc_cols_lo[0].awvalid;
    assign axi4_mc_cols_li[0].awready  = m_axi4_manycore_awready;
    assign m_axi4_manycore_awlen    = axi4_mc_cols_lo[0].awlen;
    assign m_axi4_manycore_awsize   = axi4_mc_cols_lo[0].awsize;
    assign m_axi4_manycore_awburst  = axi4_mc_cols_lo[0].awburst;
    assign m_axi4_manycore_awcache  = axi4_mc_cols_lo[0].awcache;
    assign m_axi4_manycore_awprot   = axi4_mc_cols_lo[0].awprot;
    assign m_axi4_manycore_awlock   = axi4_mc_cols_lo[0].awlock;
    assign m_axi4_manycore_awregion = 4'b0;
    assign m_axi4_manycore_awqos    = 4'b0;

    // AXI Write signals
    assign m_axi4_manycore_wdata  = axi4_mc_cols_lo[0].wdata;
    assign m_axi4_manycore_wstrb  = axi4_mc_cols_lo[0].wstrb;
    assign m_axi4_manycore_wlast  = axi4_mc_cols_lo[0].wlast;
    assign m_axi4_manycore_wvalid = axi4_mc_cols_lo[0].wvalid;
    assign axi4_mc_cols_li[0].wready = m_axi4_manycore_wready;

    // AXI Burst signals
    assign axi4_mc_cols_li[0].bid    = m_axi4_manycore_bid;
    assign axi4_mc_cols_li[0].bresp  = m_axi4_manycore_bresp;
    assign axi4_mc_cols_li[0].bvalid = m_axi4_manycore_bvalid;
    assign m_axi4_manycore_bready = axi4_mc_cols_lo[0].bready;

    // AXI Address Read signals
    assign m_axi4_manycore_arid     = axi4_mc_cols_lo[0].arid;
    assign m_axi4_manycore_araddr   = axi4_mc_cols_lo[0].araddr;
    assign m_axi4_manycore_arlen    = axi4_mc_cols_lo[0].arlen;
    assign m_axi4_manycore_arsize   = axi4_mc_cols_lo[0].arsize;
    assign m_axi4_manycore_arburst  = axi4_mc_cols_lo[0].arburst;
    assign m_axi4_manycore_arcache  = axi4_mc_cols_lo[0].arcache;
    assign m_axi4_manycore_arprot   = axi4_mc_cols_lo[0].arprot;
    assign m_axi4_manycore_arlock   = axi4_mc_cols_lo[0].arlock;
    assign m_axi4_manycore_arvalid  = axi4_mc_cols_lo[0].arvalid;
    assign axi4_mc_cols_li[0].arready  = m_axi4_manycore_arready;
    assign m_axi4_manycore_arregion = 4'b0;
    assign m_axi4_manycore_arqos    = 4'b0;

    // AXI Read signals
    assign axi4_mc_cols_li[0].rid    = m_axi4_manycore_rid;
    assign axi4_mc_cols_li[0].rdata  = m_axi4_manycore_rdata;
    assign axi4_mc_cols_li[0].rresp  = m_axi4_manycore_rresp;
    assign axi4_mc_cols_li[0].rlast  = m_axi4_manycore_rlast;
    assign axi4_mc_cols_li[0].rvalid = m_axi4_manycore_rvalid;
    assign m_axi4_manycore_rready = axi4_mc_cols_lo[0].rready;

  end : lv3_axi4_c // if (mem_cfg_p == e_vcache_blocking_axi4_f1_dram)

  else if (mem_cfg_p == e_vcache_blocking_axi4_xbar_dram ||
           mem_cfg_p == e_vcache_blocking_axi4_xbar_model) begin : lv3_xbar_c

    `declare_bsg_axi4_bus_s(1, axi_id_width_p, axi_addr_width_p, axi_data_width_p, bsg_axi4_mosi_dram_s, bsg_axi4_miso_dram_s);

    bsg_axi4_mosi_dram_s axi4_dram_bus_lo;
    bsg_axi4_miso_dram_s axi4_dram_bus_li;

    axi4_mux #(
      .slot_num_p  (num_tiles_x_p   ),
      .id_width_p  (axi_id_width_p  ),
      .addr_width_p(axi_addr_width_p),
      .data_width_p(axi_data_width_p)
    ) axi4_xbar_mux (
      .clk_i       (core_clk        ),
      .reset_i     (core_reset      ),
      .s_axi4_par_i(axi4_mc_cols_lo ),
      .s_axi4_par_o(axi4_mc_cols_li ),
      .m_axi4_ser_o(axi4_dram_bus_lo),
      .m_axi4_ser_i(axi4_dram_bus_li)
    );

    // AXI Address Write signals
    assign m_axi4_manycore_awid     = axi4_dram_bus_lo.awid;
    assign m_axi4_manycore_awaddr   = axi4_dram_bus_lo.awaddr;
    assign m_axi4_manycore_awvalid  = axi4_dram_bus_lo.awvalid;
    assign axi4_dram_bus_li.awready = m_axi4_manycore_awready;
    assign m_axi4_manycore_awlen    = axi4_dram_bus_lo.awlen;
    assign m_axi4_manycore_awsize   = axi4_dram_bus_lo.awsize;
    assign m_axi4_manycore_awburst  = axi4_dram_bus_lo.awburst;
    assign m_axi4_manycore_awcache  = axi4_dram_bus_lo.awcache;
    assign m_axi4_manycore_awprot   = axi4_dram_bus_lo.awprot;
    assign m_axi4_manycore_awlock   = axi4_dram_bus_lo.awlock;
    assign m_axi4_manycore_awregion = 4'b0;
    assign m_axi4_manycore_awqos    = 4'b0;

    // AXI Write signals
    assign m_axi4_manycore_wdata   = axi4_dram_bus_lo.wdata;
    assign m_axi4_manycore_wstrb   = axi4_dram_bus_lo.wstrb;
    assign m_axi4_manycore_wlast   = axi4_dram_bus_lo.wlast;
    assign m_axi4_manycore_wvalid  = axi4_dram_bus_lo.wvalid;
    assign axi4_dram_bus_li.wready = m_axi4_manycore_wready;

    // AXI Burst signals
    assign axi4_dram_bus_li.bid    = m_axi4_manycore_bid;
    assign axi4_dram_bus_li.bresp  = m_axi4_manycore_bresp;
    assign axi4_dram_bus_li.bvalid = m_axi4_manycore_bvalid;
    assign m_axi4_manycore_bready  = axi4_dram_bus_lo.bready;

    // AXI Address Read signals
    assign m_axi4_manycore_arid     = axi4_dram_bus_lo.arid;
    assign m_axi4_manycore_araddr   = axi4_dram_bus_lo.araddr;
    assign m_axi4_manycore_arlen    = axi4_dram_bus_lo.arlen;
    assign m_axi4_manycore_arsize   = axi4_dram_bus_lo.arsize;
    assign m_axi4_manycore_arburst  = axi4_dram_bus_lo.arburst;
    assign m_axi4_manycore_arcache  = axi4_dram_bus_lo.arcache;
    assign m_axi4_manycore_arprot   = axi4_dram_bus_lo.arprot;
    assign m_axi4_manycore_arlock   = axi4_dram_bus_lo.arlock;
    assign m_axi4_manycore_arvalid  = axi4_dram_bus_lo.arvalid;
    assign axi4_dram_bus_li.arready = m_axi4_manycore_arready;
    assign m_axi4_manycore_arregion = 4'b0;
    assign m_axi4_manycore_arqos    = 4'b0;

    // AXI Read signals
    assign axi4_dram_bus_li.rid    = m_axi4_manycore_rid;
    assign axi4_dram_bus_li.rdata  = m_axi4_manycore_rdata;
    assign axi4_dram_bus_li.rresp  = m_axi4_manycore_rresp;
    assign axi4_dram_bus_li.rlast  = m_axi4_manycore_rlast;
    assign axi4_dram_bus_li.rvalid = m_axi4_manycore_rvalid;
    assign m_axi4_manycore_rready  = axi4_dram_bus_lo.rready;

  end


`ifdef COSIM
   axi_clock_converter_v2_1_18_axi_clock_converter
     #(.C_FAMILY("virtexuplus"),
       .C_AXI_ID_WIDTH(6),
       .C_AXI_ADDR_WIDTH(64),  // Width of s_axi_awaddr, s_axi_araddr, m_axi_awaddr and
       .C_AXI_DATA_WIDTH(512), // Width of WDATA and RDATA (either side).
       .C_S_AXI_ACLK_RATIO(1), // Clock frequency ratio of SI w.r.t. MI. (Slowest of all clock inputs should have ratio=1.)
       .C_M_AXI_ACLK_RATIO(lc_core_clk_period_p/lc_clk_main_a0_p),
       // S:M or M:S must be integer ratio.
       // Format: Bit32; Range: >='h00000001.
       .C_AXI_IS_ACLK_ASYNC(1), // Indicates whether S and M clocks are asynchronous.
       // FUTURE FEATURE
       // Format: Bit1. Range = 1'b0.
       .C_AXI_PROTOCOL(0), // Protocol of this SI/MI slot.
       .C_AXI_SUPPORTS_USER_SIGNALS (0),
       .C_AXI_SUPPORTS_WRITE(1),
       .C_AXI_SUPPORTS_READ(1),
       .C_SYNCHRONIZER_STAGE(3)
       )
   axi4_dram_cdc
     (.s_axi_aclk(core_clk),
      .s_axi_aresetn(~core_reset),

      // Slave Interface Write Address Ports
      .s_axi_awid(m_axi4_manycore_awid),
      .s_axi_awaddr(m_axi4_manycore_awaddr),
      .s_axi_awlen(m_axi4_manycore_awlen),
      .s_axi_awsize(m_axi4_manycore_awsize),
      .s_axi_awburst(m_axi4_manycore_awburst),
      .s_axi_awlock(m_axi4_manycore_awlock),
      .s_axi_awcache(m_axi4_manycore_awcache),
      .s_axi_awprot(m_axi4_manycore_awprot),
      .s_axi_awregion(m_axi4_manycore_awregion),
      .s_axi_awqos(m_axi4_manycore_awqos),
      .s_axi_awvalid(m_axi4_manycore_awvalid),
      .s_axi_awready(m_axi4_manycore_awready),

      // Slave Interface Write Data Ports
      .s_axi_wid(m_axi4_manycore_wid),
      .s_axi_wdata(m_axi4_manycore_wdata),
      .s_axi_wstrb(m_axi4_manycore_wstrb),
      .s_axi_wlast(m_axi4_manycore_wlast),
      .s_axi_wvalid(m_axi4_manycore_wvalid),
      .s_axi_wready(m_axi4_manycore_wready),

      // Slave Interface Write Response Ports
      .s_axi_bid(m_axi4_manycore_bid),
      .s_axi_bresp(m_axi4_manycore_bresp),
      .s_axi_bvalid(m_axi4_manycore_bvalid),
      .s_axi_bready(m_axi4_manycore_bready),

      // Slave Interface Read Address Ports
      .s_axi_arid(m_axi4_manycore_arid),
      .s_axi_araddr(m_axi4_manycore_araddr),
      .s_axi_arlen(m_axi4_manycore_arlen),
      .s_axi_arsize(m_axi4_manycore_arsize),
      .s_axi_arburst(m_axi4_manycore_arburst),
      .s_axi_arlock(m_axi4_manycore_arlock),
      .s_axi_arcache(m_axi4_manycore_arcache),
      .s_axi_arprot(m_axi4_manycore_arprot),
      .s_axi_arregion(m_axi4_manycore_arregion),
      .s_axi_arqos(m_axi4_manycore_arqos),
      .s_axi_arvalid(m_axi4_manycore_arvalid),
      .s_axi_arready(m_axi4_manycore_arready),

      // Slave Interface Read Data Ports
      .s_axi_rid(m_axi4_manycore_rid),
      .s_axi_rdata(m_axi4_manycore_rdata),
      .s_axi_rresp(m_axi4_manycore_rresp),
      .s_axi_rlast(m_axi4_manycore_rlast),
      .s_axi_rvalid(m_axi4_manycore_rvalid),
      .s_axi_rready(m_axi4_manycore_rready),

      // Master Interface System Signals
      .m_axi_aclk(clk_main_a0),
      .m_axi_aresetn(rst_main_n),

      // Master Interface Write Address Port
      .m_axi_awid(cl_sh_ddr_awid),
      .m_axi_awaddr(cl_sh_ddr_awaddr),
      .m_axi_awlen(cl_sh_ddr_awlen),
      .m_axi_awsize(cl_sh_ddr_awsize),
      .m_axi_awburst(cl_sh_ddr_awburst),
      .m_axi_awlock(cl_sh_ddr_awlock),
      .m_axi_awcache(cl_sh_ddr_awcache),
      .m_axi_awprot(cl_sh_ddr_awprot),
      .m_axi_awregion(cl_sh_ddr_awregion),
      .m_axi_awqos(cl_sh_ddr_awqos),
      .m_axi_awvalid(cl_sh_ddr_awvalid),
      .m_axi_awready(sh_cl_ddr_awready),

      // Master Interface Write Data Ports
      .m_axi_wdata(cl_sh_ddr_wdata),
      .m_axi_wstrb(cl_sh_ddr_wstrb),
      .m_axi_wlast(cl_sh_ddr_wlast),
      .m_axi_wvalid(cl_sh_ddr_wvalid),
      .m_axi_wready(sh_cl_ddr_wready),

      // Master Interface Write Response Ports
      .m_axi_bid(sh_cl_ddr_bid),
      .m_axi_bresp(sh_cl_ddr_bresp),
      .m_axi_bvalid(sh_cl_ddr_bvalid),
      .m_axi_bready(cl_sh_ddr_bready),

      // Master Interface Read Address Port
      .m_axi_arid(cl_sh_ddr_arid),
      .m_axi_araddr(cl_sh_ddr_araddr),
      .m_axi_arlen(cl_sh_ddr_arlen),
      .m_axi_arsize(cl_sh_ddr_arsize),
      .m_axi_arburst(cl_sh_ddr_arburst),
      .m_axi_arlock(cl_sh_ddr_arlock),
      .m_axi_arcache(cl_sh_ddr_arcache),
      .m_axi_arprot(cl_sh_ddr_arprot),
      .m_axi_arregion(cl_sh_ddr_arregion),
      .m_axi_arqos(cl_sh_ddr_arqos),
      .m_axi_arvalid(cl_sh_ddr_arvalid),
      .m_axi_arready(sh_cl_ddr_arready),

      // Master Interface Read Data Ports
      .m_axi_rid(sh_cl_ddr_rid),
      .m_axi_rdata(sh_cl_ddr_rdata),
      .m_axi_rresp(sh_cl_ddr_rresp),
      .m_axi_rlast(sh_cl_ddr_rlast),
      .m_axi_rvalid(sh_cl_ddr_rvalid),
      .m_axi_rready(cl_sh_ddr_rready));
`else

   //--------------------------------------------
   // AXI4 Manycore System
   //---------------------------------------------
   assign m_axi4_manycore_rid = sh_cl_ddr_rid;
   assign m_axi4_manycore_rdata = sh_cl_ddr_rdata;
   assign m_axi4_manycore_rresp = sh_cl_ddr_rresp;
   assign m_axi4_manycore_rlast = sh_cl_ddr_rlast;
   assign m_axi4_manycore_rvalid = sh_cl_ddr_rvalid;
   assign cl_sh_ddr_rready = m_axi4_manycore_rready;

   assign cl_sh_ddr_awid = m_axi4_manycore_awid;
   assign cl_sh_ddr_awaddr = m_axi4_manycore_awaddr;
   assign cl_sh_ddr_awlen = m_axi4_manycore_awlen;
   assign cl_sh_ddr_awsize = m_axi4_manycore_awsize;
   assign cl_sh_ddr_awburst = m_axi4_manycore_awburst;
   assign cl_sh_ddr_awlock = m_axi4_manycore_awlock;
   assign cl_sh_ddr_awcache = m_axi4_manycore_awcache;
   assign cl_sh_ddr_awprot = m_axi4_manycore_awprot;
   assign cl_sh_ddr_awregion = m_axi4_manycore_awregion;
   assign cl_sh_ddr_awqos = m_axi4_manycore_awqos;
   assign cl_sh_ddr_awvalid = m_axi4_manycore_awvalid;
   assign m_axi4_manycore_awready = sh_cl_ddr_awready;

   assign cl_sh_ddr_wdata = m_axi4_manycore_wdata;
   assign cl_sh_ddr_wstrb = m_axi4_manycore_wstrb;
   assign cl_sh_ddr_wlast = m_axi4_manycore_wlast;
   assign cl_sh_ddr_wvalid = m_axi4_manycore_wvalid;
   assign m_axi4_manycore_wready = sh_cl_ddr_wready;

   assign m_axi4_manycore_bid = sh_cl_ddr_bid;
   assign m_axi4_manycore_bresp = sh_cl_ddr_bresp;
   assign m_axi4_manycore_bvalid = sh_cl_ddr_bvalid;
   assign cl_sh_ddr_bready = m_axi4_manycore_bready;

   assign cl_sh_ddr_arid = m_axi4_manycore_arid;
   assign cl_sh_ddr_araddr = m_axi4_manycore_araddr;
   assign cl_sh_ddr_arlen = m_axi4_manycore_arlen;
   assign cl_sh_ddr_arsize = m_axi4_manycore_arsize;
   assign cl_sh_ddr_arburst = m_axi4_manycore_arburst;
   assign cl_sh_ddr_arlock = m_axi4_manycore_arlock;
   assign cl_sh_ddr_arcache = m_axi4_manycore_arcache;
   assign cl_sh_ddr_arprot = m_axi4_manycore_arprot;
   assign cl_sh_ddr_arregion = m_axi4_manycore_arregion;
   assign cl_sh_ddr_arqos = m_axi4_manycore_arqos;
   assign cl_sh_ddr_arvalid = m_axi4_manycore_arvalid;
   assign m_axi4_manycore_arready = sh_cl_ddr_arready;
`endif


   //-----------------------------------------------
   // Debug bridge, used if need Virtual JTAG
   //-----------------------------------------------
`ifndef DISABLE_VJTAG_DEBUG

   // Flop for timing global clock counter
   logic [63:0]               sh_cl_glcount0_q;

   always_ff @(posedge clk_main_a0)
     if (!rst_main_n_sync)
       sh_cl_glcount0_q <= 0;
     else
       sh_cl_glcount0_q <= sh_cl_glcount0;


   // Integrated Logic Analyzers (ILA)
   ila_0 CL_ILA_0
     (
      .clk    (clk_main_a0),
      .probe0 (m_axil_ocl_awvalid)
      ,.probe1 (64'(m_axil_ocl_awaddr))
      ,.probe2 (m_axil_ocl_awready)
      ,.probe3 (m_axil_ocl_arvalid)
      ,.probe4 (64'(m_axil_ocl_araddr))
      ,.probe5 (m_axil_ocl_arready)
      );

   ila_0 CL_ILA_1
     (
      .clk    (clk_main_a0)
      ,.probe0 (m_axil_ocl_bvalid)
      ,.probe1 (sh_cl_glcount0_q)
      ,.probe2 (m_axil_ocl_bready)
      ,.probe3 (m_axil_ocl_rvalid)
      ,.probe4 ({32'b0,m_axil_ocl_rdata[31:0]})
      ,.probe5 (m_axil_ocl_rready)
      );

   // Debug Bridge
   cl_debug_bridge CL_DEBUG_BRIDGE
     (
      .clk(clk_main_a0)
      ,.S_BSCAN_drck(drck)
      ,.S_BSCAN_shift(shift)
      ,.S_BSCAN_tdi(tdi)
      ,.S_BSCAN_update(update)
      ,.S_BSCAN_sel(sel)
      ,.S_BSCAN_tdo(tdo)
      ,.S_BSCAN_tms(tms)
      ,.S_BSCAN_tck(tck)
      ,.S_BSCAN_runtest(runtest)
      ,.S_BSCAN_reset(reset)
      ,.S_BSCAN_capture(capture)
      ,.S_BSCAN_bscanid_en(bscanid_en)
      );

`endif //  `ifndef DISABLE_VJTAG_DEBUG

   // synopsys translate off
   int                        status;
   logic                      trace_en;
   initial begin
      assign trace_en = $test$plusargs("trace");
   end

   bind vanilla_core vanilla_core_trace
     #(
       .x_cord_width_p(x_cord_width_p)
       ,.y_cord_width_p(y_cord_width_p)
       ,.icache_tag_width_p(icache_tag_width_p)
       ,.icache_entries_p(icache_entries_p)
       ,.data_width_p(data_width_p)
       ,.dmem_size_p(dmem_size_p)
       )
   vtrace
     (
      .*
      ,.trace_en_i($root.tb.card.fpga.CL.trace_en)
      );


   // profilers
   //
   logic [31:0] global_ctr;

   bsg_cycle_counter global_cc
     (
      .clk_i(core_clk)
      ,.reset_i(core_reset)
      ,.ctr_r_o(global_ctr)
      );


   bind vanilla_core vanilla_core_profiler
     #(
       .x_cord_width_p(x_cord_width_p)
       ,.y_cord_width_p(y_cord_width_p)
       ,.data_width_p(data_width_p)
       ,.dmem_size_p(data_width_p)
       )
   vcore_prof
     (
      .*
      ,.global_ctr_i($root.tb.card.fpga.CL.global_ctr)
      ,.print_stat_v_i($root.tb.card.fpga.CL.hb_mc_wrapper.print_stat_v_lo)
      ,.print_stat_tag_i($root.tb.card.fpga.CL.hb_mc_wrapper.print_stat_tag_lo)
      ,.trace_en_i($root.tb.card.fpga.CL.trace_en)
      );

   // synopsys translate on

endmodule
