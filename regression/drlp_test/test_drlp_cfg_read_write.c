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

#include <bsg_manycore.h>
#include <bsg_manycore_npa.h>
#include <bsg_manycore_printing.h>
#include <inttypes.h>
#include "test_drlp_cfg_read_write.h"

#define TEST_NAME "test_drlp_cfg_read_write"

int test_drlp_cfg_read_write () {
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
	uint32_t dram_coord_x = DRAM_X;
	uint32_t dram_coord_y = DRAM_Y;
	int mismatch = 0;
	uint32_t write_data[ARRAY_LEN];
	
	for (size_t i = 0; i < ARRAY_LEN; i++)
		write_data[i] = 0X3F800000; // 1 in IEEE-754 FP

	for (size_t i = 0; i < ARRAY_LEN; i++) {
		if (i % 64 == 1)
			bsg_pr_test_info("%s: Have written %zu words to DRAM\n",
			    __func__, i);
		
		hb_mc_npa_t npa = { .x = dram_coord_x, .y = dram_coord_y, .epa = BASE_ADDR + (i*4) };
		err = hb_mc_manycore_write_mem(mc, &npa,
					       &write_data[i], sizeof(write_data[i]));
		if (err != HB_MC_SUCCESS) {
			bsg_pr_err("%s: failed to write A[%d] = 0x%08" PRIx32 " "
				   "to DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
				   __func__, i, write_data[i],
				   dram_coord_x, dram_coord_y,
				   BASE_ADDR + i);
			goto cleanup;
		}
	}

	/**************************/
	/* Writing config to DRLP */
	/**************************/
	uint32_t drlp_coord_x = DRLP_X;
	uint32_t drlp_coord_y = DRLP_Y;
	bsg_pr_test_info("Writing DRLP configure registers\n");

	int cfg_addr[DRLP_CFG_LEN] = DRLP_CFG_ADDR;
	uint32_t cfg_fp_conv1[7] = {0x47c10101, BASE_ADDR, BASE_ADDR, 0, 0, 0x00500028, 0x07000001};
	uint32_t config;

	for (size_t i = 0; i < DRLP_CFG_LEN; i++) {
		config = cfg_fp_conv1[i];
		hb_mc_npa_t npa = { .x = drlp_coord_x, .y = drlp_coord_y, .epa = cfg_addr[i] };
		err = hb_mc_manycore_write_mem(mc, &npa, &config, sizeof(config));
		if (err != HB_MC_SUCCESS) {
			bsg_pr_err("%s: failed to write to DRLP configure registers: %s\n",
				   __func__,
				   hb_mc_strerror(err));
			goto cleanup;
		}
	}
	bsg_pr_test_info("Write successful\n");

	// Turn off drlp
	config = config-1;
	hb_mc_npa_t npa = { .x = drlp_coord_x, .y = drlp_coord_y, .epa = cfg_addr[DRLP_CFG_LEN-1] };
	err = hb_mc_manycore_write_mem(mc, &npa, &config, sizeof(config));

	/******************************/
	/* Read back config from DRLP */
	/******************************/
	uint32_t read_config;
	for (size_t i = 0; i < DRLP_CFG_LEN-1; i++) {
		hb_mc_npa_t npa = { .x = drlp_coord_x, .y = drlp_coord_y, .epa = cfg_addr[i] };
		config = cfg_fp_conv1[i];
		err = hb_mc_manycore_read_mem(mc, &npa, &read_config, sizeof(read_config));
		if (err != HB_MC_SUCCESS) {
			bsg_pr_err("%s: failed to read from manycore DMEM: %s\n",
				   __func__,
				   hb_mc_strerror(err));
			goto cleanup;
		}

		bsg_pr_test_info("Completed read\n");
		if (read_config == config) {
			bsg_pr_test_info("Read back data written: 0x%08" PRIx32 "\n",
					 read_config);
		} else {
			bsg_pr_test_info("Data mismatch: read 0x%08" PRIx32 ", wrote 0x%08" PRIx32 "\n",
					 read_config, config);
		}
		r = (read_config == config ? HB_MC_SUCCESS : HB_MC_FAIL);
	}

	/*******************************/
	/* Read back results from DRAM */
	/*******************************/
	uint32_t read_data;
	for (size_t i = 0; i < 16; i++) {
		hb_mc_npa_t npa = { .x = dram_coord_x, .y = dram_coord_y , .epa = BASE_ADDR + (i*4) };
		err = hb_mc_manycore_read_mem(mc, &npa,
					      &read_data, sizeof(read_data));
		if (err != HB_MC_SUCCESS) {
			bsg_pr_err("%s: failed to read A[%d] "
				   "from DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
				   i, dram_coord_x, dram_coord_y, BASE_ADDR + i);
			goto cleanup;
		}

		if (read_data == 0x41880000) {
			bsg_pr_test_info("Correct! Read result %x \n", read_data);
		}
		else {
			bsg_pr_test_info("Wrong! Read result %x, should be 0x41880000 \n", read_data);
		}
		r = (read_data == 0x41880000 ? HB_MC_SUCCESS : HB_MC_FAIL);
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
	int rc = test_drlp_cfg_read_write();
	*exit_code = rc;
	bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
	return;
}
#else
int main(int argc, char ** argv) {
	bsg_pr_test_info(TEST_NAME " Regression Test (F1)\n");
	int rc = test_drlp_cfg_read_write();
	bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
	return rc;
}
#endif
