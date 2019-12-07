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
#include <stdlib.h>
#include "test_drlp_fp.h"

#define TEST_NAME "test_drlp_fp"

static const uint32_t drlp_coord_x = DRLP_X;
static const uint32_t drlp_coord_y = DRLP_Y;
static const uint32_t dram_coord_x = DRAM_X;
static const uint32_t dram_coord_y = DRAM_Y;
static const uint32_t cfg_addr[DRLP_CFG_LEN] = DRLP_CFG_ADDR;

int write_configure(hb_mc_manycore_t *mc, uint32_t config_array[DRLP_CFG_LEN]) {
	uint32_t config;
	int err;
	for (size_t i = 0; i < DRLP_CFG_LEN; i++) {
		config = config_array[i];
		hb_mc_npa_t npa = { .x = drlp_coord_x, .y = drlp_coord_y, .epa = cfg_addr[i] };
		err = hb_mc_manycore_write_mem(mc, &npa, &config, sizeof(config));
		if (err != HB_MC_SUCCESS) {
			bsg_pr_err("%s: failed to write to DRLP configure registers: %s\n", __func__, hb_mc_strerror(err));
			hb_mc_manycore_exit(mc);
			return err;
		}
	}
	bsg_pr_test_info("Write configure successful\n");
	// Turn off drlp
	config = config-1;
	hb_mc_npa_t npa = { .x = drlp_coord_x, .y = drlp_coord_y, .epa = cfg_addr[DRLP_CFG_LEN-1] };
	err = hb_mc_manycore_write_mem(mc, &npa, &config, sizeof(config));
	if (err != HB_MC_SUCCESS) {
		bsg_pr_err("%s: failed to write to DRLP configure registers: %s\n", __func__, hb_mc_strerror(err));
		hb_mc_manycore_exit(mc);
	}
	return err;
}

int read_configure(hb_mc_manycore_t *mc, uint32_t config_array[DRLP_CFG_LEN]) {
	/******************************/
	/* Read back config from DRLP */
	/******************************/
	int err;
	uint32_t read_config;
	uint32_t config;
	for (size_t i = 0; i < DRLP_CFG_LEN; i++) {
		hb_mc_npa_t npa = { .x = drlp_coord_x, .y = drlp_coord_y, .epa = cfg_addr[i] };
		config = config_array[i];
		err = hb_mc_manycore_read_mem(mc, &npa, &read_config, sizeof(read_config));
		if (err != HB_MC_SUCCESS) {
			bsg_pr_err("%s: failed to read from manycore DMEM: %s\n", __func__, hb_mc_strerror(err));
			hb_mc_manycore_exit(mc);
			return err;
		}
		if (read_config == config) {
			bsg_pr_test_info("Read back data written: 0x%08" PRIx32 "\n", read_config);
		} else {
			bsg_pr_test_info("Data mismatch: read 0x%08" PRIx32 ", wrote 0x%08" PRIx32 "\n", read_config, config);
		}
		err = (read_config == config ? HB_MC_SUCCESS : HB_MC_FAIL);
	}
	return err;
}

int write_file (hb_mc_manycore_t *mc, FILE *f, uint32_t base_addr, int file_format) {
	// file_format: 2->binary file, 16-> hex file
	if (f == NULL) {
		bsg_pr_err ("Error: No such file\n");
		return HB_MC_FAIL;
	}
	int err, i=0;
	char num[34];
	uint32_t num_int;
	while(fgets(num, sizeof(num), f)) {
		if (i % 1024 == 1)
			bsg_pr_test_info("%s: Have written %zu words to DRAM\n", __func__, i);

		num_int = (int)strtol(num, NULL,file_format);
		hb_mc_npa_t npa = { .x = dram_coord_x, .y = dram_coord_y, .epa = base_addr + (i*4) };
		i++;
		err = hb_mc_manycore_write_mem(mc, &npa, &num_int, sizeof(num_int));
		if (err != HB_MC_SUCCESS) {
			bsg_pr_err("%s: failed to write A[%d] = 0x%08" PRIx32 " "
				   "to DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
				   __func__, i, num_int,
				   dram_coord_x, dram_coord_y,
				   base_addr + i);
			hb_mc_manycore_exit(mc);
			return err;
		}
	}
	bsg_pr_test_info("Write file done!\n");
	return err;
}

int conv1_fp (hb_mc_manycore_t *mc) {
	int err = HB_MC_SUCCESS;
	uint32_t config[DRLP_CFG_LEN] = {0x73D30413, CONV1_ACT_ADDR, CONV1_WGT_ADDR, RMEM_ADDR0, 0, 0xDC800039, 0x06002031};
	uint32_t done = 0;
	for (int k = 0; k < 8; ++k){
		err = write_configure(mc, config);
		if (err != HB_MC_SUCCESS)
			return err;
		// Wait for stop
		hb_mc_npa_t done_npa = { .x = drlp_coord_x, .y = drlp_coord_y, .epa = DRLP_DONE_ADDR };
		while (done != 1) {
			for (int i=0; i<999; i++){}
			hb_mc_manycore_read_mem(mc, &done_npa, &done, sizeof(done));
		}
		bsg_pr_test_info("CONV1 %d/8 DONE\n", k+1);
		done = 0;
		if (k==0 || k==4) {
			config[2] = config[2] + (1024+16);
			config[5] = 0xdc80011d;
		} else if (k==1 || k==5) {
			config[2] = config[2] + 1024;
			config[5] = 0xdc80151d;
		} else if (k==2 || k==6) {
			config[2] = config[2] + 1024;
			config[5] = 0xdc80161c;
		} else if (k==3) {
			config[2] = config[2] + 1024;
			config[3] = config[3] + 6400;
			config[5] = 0xdc800019;
		}
	}
	return err;
}

int conv2_fp (hb_mc_manycore_t *mc) {
	int err = HB_MC_SUCCESS;
	uint32_t config[DRLP_CFG_LEN] = {0x6BC82008, RMEM_ADDR0, CONV2_WGT_ADDR, RMEM_ADDR1, 0, 0xB4000028, 0x14008051};
	uint32_t done = 0;
	for (int k = 0; k < 4; ++k){
		err = write_configure(mc, config);
		if (err != HB_MC_SUCCESS)
			return err;
		// Wait for stop
		hb_mc_npa_t done_npa = { .x = drlp_coord_x, .y = drlp_coord_y, .epa = DRLP_DONE_ADDR };
		while (done != 1) {
			for (int i=0; i<999; i++){}
			hb_mc_manycore_read_mem(mc, &done_npa, &done, sizeof(done));
		}
		bsg_pr_test_info("CONV2 %d/4 DONE\n", k+1);
		done = 0;
		config[2] = config[2] + 8208;
		config[3] = config[3] + 1296;
		if (k==0) {
			config[5] = 0xB4000008;
		} 
	}
	return err;
}

int conv3_fp (hb_mc_manycore_t *mc) {
	int err = HB_MC_SUCCESS;
	uint32_t config[DRLP_CFG_LEN] = {0x27C62006, RMEM_ADDR1, CONV3_WGT_ADDR, RMEM_ADDR2, 0, 0x3F000028, 0x24009051};
	uint32_t done = 0;
	for (int k = 0; k < 4; ++k){
		err = write_configure(mc, config);
		if (err != HB_MC_SUCCESS)
			return err;
		// Wait for stop
		hb_mc_npa_t done_npa = { .x = drlp_coord_x, .y = drlp_coord_y, .epa = DRLP_DONE_ADDR };
		while (done != 1) {
			for (int i=0; i<999; i++){}
			hb_mc_manycore_read_mem(mc, &done_npa, &done, sizeof(done));
		}
		bsg_pr_test_info("CONV3 %d/4 DONE\n", k+1);
		done = 0;
		config[2] = config[2] + 9232;
		config[3] = config[3] + 784;
		if (k==0) {
			config[5] = 0x3F000008;
		} 
	}
	return err;
}

int fc1_fp (hb_mc_manycore_t *mc) {
	int err = HB_MC_SUCCESS;
	/* uint32_t config[DRLP_CFG_LEN] = {0xA7C00201, 0, FC1_WGT_ADDR, RMEM_ADDR3, 0, 0x300028, 0x36000001}; */
	uint32_t config[DRLP_CFG_LEN] = {0xA7C020AE, RMEM_ADDR2, FC1_WGT_ADDR, RMEM_ADDR3, 0, 0x10680028, 0x34004C03};
	uint32_t done = 0;
	err = write_configure(mc, config);
	// Wait for stop
	hb_mc_npa_t done_npa = { .x = drlp_coord_x, .y = drlp_coord_y, .epa = DRLP_DONE_ADDR };
	while (done != 1) {
		for (int i=0; i<999; i++){}
		hb_mc_manycore_read_mem(mc, &done_npa, &done, sizeof(done));
	}
	bsg_pr_test_info("FC1 DONE\n");
	return err;
}

int fc2_fp (hb_mc_manycore_t *mc) {
	int err = HB_MC_SUCCESS;
	uint32_t config[DRLP_CFG_LEN] = {0x84C0011C, RMEM_ADDR3, FC2_WGT_ADDR, RESULT_ADDR, 0, 0x02B80028, 0x4500009B};
	uint32_t done = 0;
	err = write_configure(mc, config);
	// Wait for stop
	hb_mc_npa_t done_npa = { .x = drlp_coord_x, .y = drlp_coord_y, .epa = DRLP_DONE_ADDR };
	while (done != 1) {
		for (int i=0; i<999; i++){}
		hb_mc_manycore_read_mem(mc, &done_npa, &done, sizeof(done));
	}
	bsg_pr_test_info("FC2 DONE\n");
	return err;
}


int test_drlp_fp () {
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
	// Write image
	FILE *f = fopen("/mnt/users/ssd1/homes/huwan/drlp_software/debug_weights/dram.vec", "r");
	write_file(mc, f, CONV1_ACT_ADDR*4, 2);
	fclose(f);

	// CONV1
	f = fopen("/mnt/users/ssd1/homes/huwan/drlp_software/debug_weights/conv1_wgt.vec", "r");
	write_file(mc, f, CONV1_WGT_ADDR*4, 2);
	fclose(f);
	conv1_fp(mc);

	// CONV2
	f = fopen("/mnt/users/ssd1/homes/huwan/drlp_software/debug_weights/conv2_wgt.vec", "r");
	write_file(mc, f, CONV2_WGT_ADDR*4, 2);
	fclose(f);
	conv2_fp(mc);

	// CONV3
	f = fopen("/mnt/users/ssd1/homes/huwan/drlp_software/debug_weights/conv3_wgt.vec", "r");
	write_file(mc, f, CONV3_WGT_ADDR*4, 2);
	fclose(f);
	conv3_fp(mc);

	// FC1 
	f = fopen("/mnt/users/ssd1/homes/huwan/drlp_software/debug_weights/fc1_wgt.vec", "r");
	write_file(mc, f, FC1_WGT_ADDR*4, 2);
	fclose(f);
	fc1_fp(mc);

	// FC2
	f = fopen("/mnt/users/ssd1/homes/huwan/drlp_software/debug_weights/fc2_wgt.vec", "r");
	write_file(mc, f, FC2_WGT_ADDR*4, 2);
	fclose(f);
	fc2_fp(mc);

	/*******************************/
	/* Read back results from DRAM */
	/*******************************/
	uint32_t read_data;
	for (size_t i = 0; i < 16; i++) {
		hb_mc_npa_t npa = { .x = dram_coord_x, .y = dram_coord_y , .epa = RESULT_ADDR*4 + (i*4) };
		err = hb_mc_manycore_read_mem(mc, &npa,
					      &read_data, sizeof(read_data));
		if (err != HB_MC_SUCCESS) {
			bsg_pr_err("%s: failed to read A[%d] "
				   "from DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
				   i, dram_coord_x, dram_coord_y, RMEM_ADDR0 + i);
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
	int rc = test_drlp_fp();
	*exit_code = rc;
	bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
	return;
}
#else
int main(int argc, char ** argv) {
	bsg_pr_test_info(TEST_NAME " Regression Test (F1)\n");
	int rc = test_drlp_fp();
	bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
	return rc;
}
#endif
