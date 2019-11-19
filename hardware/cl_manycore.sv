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
  import bsg_mem_cfg_pkg::*;
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
logic [5:0] m_axi4_manycore_awid;
logic [63:0] m_axi4_manycore_awaddr;
logic [7:0] m_axi4_manycore_awlen;
logic [2:0] m_axi4_manycore_awsize;
logic [1:0] m_axi4_manycore_awburst;
logic [0:0] m_axi4_manycore_awlock;
logic [3:0] m_axi4_manycore_awcache;
logic [2:0] m_axi4_manycore_awprot;
logic [3:0] m_axi4_manycore_awregion;
logic [3:0] m_axi4_manycore_awqos;
logic m_axi4_manycore_awvalid;
logic m_axi4_manycore_awready;

logic [511:0] m_axi4_manycore_wdata;
logic [63:0] m_axi4_manycore_wstrb;
logic m_axi4_manycore_wlast;
logic m_axi4_manycore_wvalid;
logic m_axi4_manycore_wready;

logic [5:0] m_axi4_manycore_bid;
logic [1:0] m_axi4_manycore_bresp;
logic m_axi4_manycore_bvalid;
logic m_axi4_manycore_bready;

logic [5:0] m_axi4_manycore_arid;
logic [63:0] m_axi4_manycore_araddr;
logic [7:0] m_axi4_manycore_arlen;
logic [2:0] m_axi4_manycore_arsize;
logic [1:0] m_axi4_manycore_arburst;
logic [0:0] m_axi4_manycore_arlock;
logic [3:0] m_axi4_manycore_arcache;
logic [2:0] m_axi4_manycore_arprot;
logic [3:0] m_axi4_manycore_arregion;
logic [3:0] m_axi4_manycore_arqos;
logic m_axi4_manycore_arvalid;
logic m_axi4_manycore_arready;

logic [5:0] m_axi4_manycore_rid;
logic [511:0] m_axi4_manycore_rdata;
logic [1:0] m_axi4_manycore_rresp;
logic m_axi4_manycore_rlast;
logic m_axi4_manycore_rvalid;
logic m_axi4_manycore_rready;

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

//--------------------------------------------
// AXI-Lite OCL System
//---------------------------------------------
axi_register_slice_light AXIL_OCL_REG_SLC (
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

// manycore wrapper
`declare_bsg_manycore_link_sif_s(addr_width_p, data_width_p, x_cord_width_p, y_cord_width_p, load_id_width_p);

bsg_manycore_link_sif_s [num_cache_p-1:0] cache_link_sif_li;
bsg_manycore_link_sif_s [num_cache_p-1:0] cache_link_sif_lo;

logic [num_cache_p-1:0][x_cord_width_p-1:0] cache_x_lo;
logic [num_cache_p-1:0][y_cord_width_p-1:0] cache_y_lo;

bsg_manycore_link_sif_s loader_link_sif_li;
bsg_manycore_link_sif_s loader_link_sif_lo;


bsg_manycore_wrapper #(
  .addr_width_p(addr_width_p)
  ,.data_width_p(data_width_p)
  ,.num_tiles_x_p(num_tiles_x_p)
  ,.num_tiles_y_p(num_tiles_y_p)
  ,.hetero_type_vec_p(hetero_type_vec_p)
  ,.dmem_size_p(dmem_size_p)
  ,.icache_entries_p(icache_entries_p)
  ,.icache_tag_width_p(icache_tag_width_p)
  ,.epa_byte_addr_width_p(epa_byte_addr_width_p)
  ,.dram_ch_addr_width_p(dram_ch_addr_width_p)
  ,.load_id_width_p(load_id_width_p)
  ,.num_cache_p(num_cache_p)
  ,.vcache_size_p(vcache_size_p)
  ,.vcache_block_size_in_words_p(block_size_in_words_p)
  ,.vcache_sets_p(sets_p)
  ,.branch_trace_en_p(branch_trace_en_p)
) manycore_wrapper (
  .clk_i(clk_main_a0)
  ,.reset_i(~rst_main_n_sync)

  ,.cache_link_sif_i(cache_link_sif_li)
  ,.cache_link_sif_o(cache_link_sif_lo)

  ,.cache_x_o(cache_x_lo)
  ,.cache_y_o(cache_y_lo)

  ,.loader_link_sif_i(loader_link_sif_li)
  ,.loader_link_sif_o(loader_link_sif_lo)
);


// configurable memory system
//
memory_system #(
  .mem_cfg_p(mem_cfg_p)
  
  ,.bsg_global_x_p(num_tiles_x_p)
  ,.bsg_global_y_p(num_tiles_y_p)

  ,.data_width_p(data_width_p)
  ,.addr_width_p(addr_width_p)
  ,.x_cord_width_p(x_cord_width_p)
  ,.y_cord_width_p(y_cord_width_p)
  ,.load_id_width_p(load_id_width_p)

  ,.block_size_in_words_p(block_size_in_words_p)
  ,.sets_p(sets_p)
  ,.ways_p(ways_p)

  ,.axi_id_width_p(axi_id_width_p)
  ,.axi_addr_width_p(axi_addr_width_p)
  ,.axi_data_width_p(axi_data_width_p)
  ,.axi_burst_len_p(axi_burst_len_p)

) memsys (
  .clk_i(clk_main_a0)
  ,.reset_i(~rst_main_n_sync)

  ,.link_sif_i(cache_link_sif_lo)
  ,.link_sif_o(cache_link_sif_li)

  ,.axi_awid_o    (m_axi4_manycore_awid)
  ,.axi_awaddr_o  (m_axi4_manycore_awaddr)
  ,.axi_awlen_o   (m_axi4_manycore_awlen)
  ,.axi_awsize_o  (m_axi4_manycore_awsize)
  ,.axi_awburst_o (m_axi4_manycore_awburst)
  ,.axi_awcache_o (m_axi4_manycore_awcache)
  ,.axi_awprot_o  (m_axi4_manycore_awprot)
  ,.axi_awlock_o  (m_axi4_manycore_awlock)
  ,.axi_awvalid_o (m_axi4_manycore_awvalid)
  ,.axi_awready_i (m_axi4_manycore_awready)

  ,.axi_wdata_o   (m_axi4_manycore_wdata)
  ,.axi_wstrb_o   (m_axi4_manycore_wstrb)
  ,.axi_wlast_o   (m_axi4_manycore_wlast)
  ,.axi_wvalid_o  (m_axi4_manycore_wvalid)
  ,.axi_wready_i  (m_axi4_manycore_wready)

  ,.axi_bid_i     (m_axi4_manycore_bid)
  ,.axi_bresp_i   (m_axi4_manycore_bresp)
  ,.axi_bvalid_i  (m_axi4_manycore_bvalid)
  ,.axi_bready_o  (m_axi4_manycore_bready)

  ,.axi_arid_o    (m_axi4_manycore_arid)
  ,.axi_araddr_o  (m_axi4_manycore_araddr)
  ,.axi_arlen_o   (m_axi4_manycore_arlen)
  ,.axi_arsize_o  (m_axi4_manycore_arsize)
  ,.axi_arburst_o (m_axi4_manycore_arburst)
  ,.axi_arcache_o (m_axi4_manycore_arcache)
  ,.axi_arprot_o  (m_axi4_manycore_arprot)
  ,.axi_arlock_o  (m_axi4_manycore_arlock)
  ,.axi_arvalid_o (m_axi4_manycore_arvalid)
  ,.axi_arready_i (m_axi4_manycore_arready)

  ,.axi_rid_i     (m_axi4_manycore_rid)
  ,.axi_rdata_i   (m_axi4_manycore_rdata)
  ,.axi_rresp_i   (m_axi4_manycore_rresp)
  ,.axi_rlast_i   (m_axi4_manycore_rlast)
  ,.axi_rvalid_i  (m_axi4_manycore_rvalid)
  ,.axi_rready_o  (m_axi4_manycore_rready)
);

assign m_axi4_manycore_awregion = 4'b0;
assign m_axi4_manycore_awqos = 4'b0;

assign m_axi4_manycore_arregion = 4'b0;
assign m_axi4_manycore_arqos = 4'b0;




// manycore link

logic [x_cord_width_p-1:0] mcl_x_cord_lp = '0;
logic [y_cord_width_p-1:0] mcl_y_cord_lp = '0;

logic print_stat_v_lo;
logic [data_width_p-1:0] print_stat_tag_lo;

axil_to_mcl #(
  .num_mcl_p        (1                )
  ,.num_tiles_x_p    (num_tiles_x_p    )
  ,.num_tiles_y_p    (num_tiles_y_p    )
  ,.addr_width_p     (addr_width_p     )
  ,.data_width_p     (data_width_p     )
  ,.x_cord_width_p   (x_cord_width_p   )
  ,.y_cord_width_p   (y_cord_width_p   )
  ,.load_id_width_p  (load_id_width_p  )
  ,.max_out_credits_p(max_out_credits_p)
) axil_to_mcl_inst (
  .clk_i             (clk_main_a0       )
  ,.reset_i           (~rst_main_n_sync  )

  // axil slave interface
  ,.s_axil_mcl_awvalid(m_axil_ocl_awvalid)
  ,.s_axil_mcl_awaddr (m_axil_ocl_awaddr )
  ,.s_axil_mcl_awready(m_axil_ocl_awready)
  ,.s_axil_mcl_wvalid (m_axil_ocl_wvalid )
  ,.s_axil_mcl_wdata  (m_axil_ocl_wdata  )
  ,.s_axil_mcl_wstrb  (m_axil_ocl_wstrb  )
  ,.s_axil_mcl_wready (m_axil_ocl_wready )
  ,.s_axil_mcl_bresp  (m_axil_ocl_bresp  )
  ,.s_axil_mcl_bvalid (m_axil_ocl_bvalid )
  ,.s_axil_mcl_bready (m_axil_ocl_bready )
  ,.s_axil_mcl_araddr (m_axil_ocl_araddr )
  ,.s_axil_mcl_arvalid(m_axil_ocl_arvalid)
  ,.s_axil_mcl_arready(m_axil_ocl_arready)
  ,.s_axil_mcl_rdata  (m_axil_ocl_rdata  )
  ,.s_axil_mcl_rresp  (m_axil_ocl_rresp  )
  ,.s_axil_mcl_rvalid (m_axil_ocl_rvalid )
  ,.s_axil_mcl_rready (m_axil_ocl_rready )

  // manycore link
  ,.link_sif_i        (loader_link_sif_lo)
  ,.link_sif_o        (loader_link_sif_li)
  ,.my_x_i            (mcl_x_cord_lp     )
  ,.my_y_i            (mcl_y_cord_lp     )

  ,.print_stat_v_o(print_stat_v_lo)
  ,.print_stat_tag_o(print_stat_tag_lo)
);

//-----------------------------------------------
// Debug bridge, used if need Virtual JTAG
//-----------------------------------------------
`ifndef DISABLE_VJTAG_DEBUG

// Flop for timing global clock counter
logic[63:0] sh_cl_glcount0_q;

always_ff @(posedge clk_main_a0)
   if (!rst_main_n_sync)
      sh_cl_glcount0_q <= 0;
   else
      sh_cl_glcount0_q <= sh_cl_glcount0;


// Integrated Logic Analyzers (ILA)
ila_0 CL_ILA_0 (
                .clk    (clk_main_a0),
                .probe0 (m_axil_ocl_awvalid)
                ,.probe1 (64'(m_axil_ocl_awaddr))
                ,.probe2 (m_axil_ocl_awready)
                ,.probe3 (m_axil_ocl_arvalid)
                ,.probe4 (64'(m_axil_ocl_araddr))
                ,.probe5 (m_axil_ocl_arready)
                );

 ila_0 CL_ILA_1 (
                .clk    (clk_main_a0)
                ,.probe0 (m_axil_ocl_bvalid)
                ,.probe1 (sh_cl_glcount0_q)
                ,.probe2 (m_axil_ocl_bready)
                ,.probe3 (m_axil_ocl_rvalid)
                ,.probe4 ({32'b0,m_axil_ocl_rdata[31:0]})
                ,.probe5 (m_axil_ocl_rready)
                 );

// Debug Bridge
cl_debug_bridge CL_DEBUG_BRIDGE (
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

bind vanilla_core vanilla_core_trace #(
  .x_cord_width_p(x_cord_width_p)
  ,.y_cord_width_p(y_cord_width_p)
  ,.icache_tag_width_p(icache_tag_width_p)
  ,.icache_entries_p(icache_entries_p)
  ,.data_width_p(data_width_p)
  ,.dmem_size_p(dmem_size_p)
) vtrace (
  .*
  ,.trace_en_i(1'b1)
);


// profilers
//
logic [31:0] global_ctr;

bsg_cycle_counter global_cc (
  .clk_i(clk_main_a0)
  ,.reset_i(~rst_main_n_sync)
  ,.ctr_r_o(global_ctr)
);


bind vanilla_core vanilla_core_profiler #(
  .x_cord_width_p(x_cord_width_p)
  ,.y_cord_width_p(y_cord_width_p)
  ,.data_width_p(data_width_p)
  ,.dmem_size_p(data_width_p)
) vcore_prof (
  .*
  ,.global_ctr_i($root.tb.card.fpga.CL.global_ctr)
  ,.print_stat_v_i($root.tb.card.fpga.CL.print_stat_v_lo)
  ,.print_stat_tag_i($root.tb.card.fpga.CL.print_stat_tag_lo)
  ,.trace_en_i(1'b1)
);

if (mem_cfg_p == e_mem_cfg_default) begin

  bind bsg_cache vcache_profiler #(
    .data_width_p(data_width_p)
  ) vcache_prof (
    .*
    ,.global_ctr_i($root.tb.card.fpga.CL.global_ctr)
    ,.print_stat_v_i($root.tb.card.fpga.CL.print_stat_v_lo)
    ,.print_stat_tag_i($root.tb.card.fpga.CL.print_stat_tag_lo)
  );

end
else if (mem_cfg_p == e_mem_cfg_infinite) begin
  bind bsg_nonsynth_mem_infinite infinite_mem_profiler #(
    .data_width_p(data_width_p)
    ,.x_cord_width_p(x_cord_width_p)
    ,.y_cord_width_p(y_cord_width_p)
  ) infty_mem_prof (
    .*
    ,.global_ctr_i($root.tb.card.fpga.CL.global_ctr)
    ,.print_stat_v_i($root.tb.card.fpga.CL.print_stat_v_lo)
    ,.print_stat_tag_i($root.tb.card.fpga.CL.print_stat_tag_lo)
  );
end

// synopsys translate on

endmodule
