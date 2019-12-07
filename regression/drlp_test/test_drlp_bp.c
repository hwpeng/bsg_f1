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

#include "test_drlp_defines.h"
#include "test_drlp_libs.h"

#define TEST_NAME "test_drlp_bp"


int test_drlp_bp () {
	hb_mc_manycore_t manycore = {0}, *mc = &manycore;
	int err, r = HB_MC_FAIL;

	/********/
	/* INIT */
	/********/
	err = hb_mc_manycore_init(mc, TEST_NAME, 0);
	if (err != HB_MC_SUCCESS) {
		bsg_pr_err("%s: failed to initialize manycore: %s\n",
			   __func__, hb_mc_strerror(err));
		return HB_MC_FAIL;
	}

	/************************/
	/* Writing data to DRAM */
	/************************/
	// Write out gd
	FILE *f = fopen("/mnt/users/ssd1/homes/huwan/drlp_software/debug_weights/out_gd.vec", "r");
	write_file(mc, f, OUT_GD_ADDR*4, 2);
	fclose(f);

	// FC2_dW
	fc2_dw(mc);

	// FC2_dX 
	f = fopen("/mnt/users/ssd1/homes/huwan/drlp_software/debug_weights/fc2_wgt_bp.vec", "r");
	write_file(mc, f, FC2BP_WGT_ADDR*4, 2);
	fclose(f);
	fc2_dx(mc);

	// FC1_dW
	fc1_dw(mc);

	// FC1_dX 
	f = fopen("/mnt/users/ssd1/homes/huwan/drlp_software/debug_weights/fc1_wgt_bp.vec", "r");
	write_file(mc, f, FC1BP_WGT_ADDR*4, 2);
	fclose(f);
	fc1_dx(mc);

	// CONV3_dW
	conv3_dw(mc);

	// CONV3_dX 
	f = fopen("/mnt/users/ssd1/homes/huwan/drlp_software/debug_weights/conv3_wgt_bp.vec", "r");
	write_file(mc, f, CONV3BP_WGT_ADDR*4, 2);
	fclose(f);
	conv3_dx(mc);

	// CONV2_dW
	conv2_dw(mc);

	// CONV2_dX 
	f = fopen("/mnt/users/ssd1/homes/huwan/drlp_software/debug_weights/conv2_wgt_bp.vec", "r");
	write_file(mc, f, CONV2BP_WGT_ADDR*4, 2);
	fclose(f);
	conv2_dx(mc);

	// CONV1_dW
	f = fopen("/mnt/users/ssd1/homes/huwan/drlp_software/debug_weights/conv1_act_bp.vec", "r");
	write_file(mc, f, CONV1BP_ACT_ADDR*4, 2);
	fclose(f);
	conv1_dw(mc);

	/*******************************/
	/* Read back results from DRAM */
	/*******************************/
	uint32_t read_data;
	uint32_t read_addr;
	bsg_pr_test_info("Read FC2_DW \n");
	read_addr = FC2BP_DW_ADDR;
	for (size_t i = 0; i < 16; i++) {
		hb_mc_npa_t npa = { .x = dram_coord_x, .y = dram_coord_y , .epa = read_addr*4 + (i*4) };
		err = hb_mc_manycore_read_mem(mc, &npa,
					      &read_data, sizeof(read_data));
		if (err != HB_MC_SUCCESS) {
			bsg_pr_err("%s: failed to read A[%d] "
				   "from DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
				   i, dram_coord_x, dram_coord_y, read_addr + i);
			goto cleanup;
		}
		bsg_pr_test_info("Read result(%d) %x \n", i, read_data);
	}
	bsg_pr_test_info("Read FC1_DW \n");
	read_addr = FC1BP_DW_ADDR;
	for (size_t i = 0; i < 16; i++) {
		hb_mc_npa_t npa = { .x = dram_coord_x, .y = dram_coord_y , .epa = read_addr*4 + (i*4) };
		err = hb_mc_manycore_read_mem(mc, &npa,
					      &read_data, sizeof(read_data));
		if (err != HB_MC_SUCCESS) {
			bsg_pr_err("%s: failed to read A[%d] "
				   "from DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
				   i, dram_coord_x, dram_coord_y, read_addr + i);
			goto cleanup;
		}
		bsg_pr_test_info("Read result(%d) %x \n", i, read_data);
	}
	bsg_pr_test_info("Read CONV3_DW \n");
	read_addr = CONV3BP_DW_ADDR;
	for (size_t i = 0; i < 16; i++) {
		hb_mc_npa_t npa = { .x = dram_coord_x, .y = dram_coord_y , .epa = read_addr*4 + (i*4) };
		err = hb_mc_manycore_read_mem(mc, &npa,
					      &read_data, sizeof(read_data));
		if (err != HB_MC_SUCCESS) {
			bsg_pr_err("%s: failed to read A[%d] "
				   "from DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
				   i, dram_coord_x, dram_coord_y, read_addr + i);
			goto cleanup;
		}
		bsg_pr_test_info("Read result(%d) %x \n", i, read_data);
	}
	bsg_pr_test_info("Read CONV2_DW \n");
	read_addr = CONV2BP_DW_ADDR;
	for (size_t i = 0; i < 16; i++) {
		hb_mc_npa_t npa = { .x = dram_coord_x, .y = dram_coord_y , .epa = read_addr*4 + (i*4) };
		err = hb_mc_manycore_read_mem(mc, &npa,
					      &read_data, sizeof(read_data));
		if (err != HB_MC_SUCCESS) {
			bsg_pr_err("%s: failed to read A[%d] "
				   "from DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
				   i, dram_coord_x, dram_coord_y, read_addr + i);
			goto cleanup;
		}
		bsg_pr_test_info("Read result(%d) %x \n", i, read_data);
	}
	bsg_pr_test_info("Read CONV1_DW \n");
	read_addr = CONV1BP_DW_ADDR;
	for (size_t i = 0; i < 16; i++) {
		hb_mc_npa_t npa = { .x = dram_coord_x, .y = dram_coord_y , .epa = read_addr*4 + (i*4) };
		err = hb_mc_manycore_read_mem(mc, &npa,
					      &read_data, sizeof(read_data));
		if (err != HB_MC_SUCCESS) {
			bsg_pr_err("%s: failed to read A[%d] "
				   "from DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
				   i, dram_coord_x, dram_coord_y, read_addr + i);
			goto cleanup;
		}
		bsg_pr_test_info("Read result(%d) %x \n", i, read_data);
	}
	/*******/
	/* END */
	/*******/
cleanup:
	hb_mc_manycore_exit(mc);
	return r;
}

#ifdef COSIM
void cosim_main(uint32_t *exit_code, char * args) {
	// We aren't passed command line arguments directly so we parse them
	// from *args. args is a string from VCS - to pass a string of arguments
	// to args, pass c_args to VCS as follows: +c_args="<space separated
	// list of args>"
	int argc = get_argc(args);
	char *argv[argc];
	get_argv(args, argc, argv);

#ifdef VCS
	svScope scope;
	scope = svGetScopeFromName("tb");
	svSetScope(scope);
#endif
	bsg_pr_test_info(TEST_NAME " Regression Test (COSIMULATION)\n");
	int rc = test_drlp_bp();
	*exit_code = rc;
	bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
	return;
}
#else
int main(int argc, char ** argv) {
	bsg_pr_test_info(TEST_NAME " Regression Test (F1)\n");
	int rc = test_drlp_bp();
	bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
	return rc;
}
#endif
