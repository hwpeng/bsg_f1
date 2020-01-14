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

#ifndef __LIBRARY_TESTS_H
#define __LIBRARY_TESTS_H

#include <bsg_manycore_loader.h>
#include <bsg_manycore_errno.h>	
#include <bsg_manycore_printing.h>

#include "../cl_manycore_regression.h"

#endif 

#pragma once

#include <stdio.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <bsg_manycore.h>
#include <bsg_manycore_npa.h>
#include <bsg_manycore_printing.h>
#include <inttypes.h>


#define DRLP_X 3 
#define DRLP_Y 4
#define DRAM_X 3
#define DRAM_Y 5

#define DRLP_CFG_LEN  7
#define DRLP_CFG_ADDR {0x0000, 0x0004, 0x0008, 0x000C, 0x0010, 0x0014, 0x0018, 0x001C}
#define DRLP_DONE_ADDR 0x0020
#define DRLP_RMEM_PREFIX 0x8000

#define CONV1_ACT_ADDR 0
#define CONV1_WGT_ADDR (29*1024)
#define CONV2_WGT_ADDR (38*1024)
#define CONV3_WGT_ADDR (71*1024)
#define FC1_WGT_ADDR (108*1024)
#define FC2_WGT_ADDR (1722*1024)
#define OUT_GD_ADDR (1725*1024)
#define FC2BP_WGT_ADDR (1726*1024)
#define FC1BP_WGT_ADDR (1736*1024)
#define CONV3BP_WGT_ADDR (3611*1024)
#define CONV2BP_WGT_ADDR (3648*1024)
#define CONV1BP_ACT_ADDR (3681*1024)
#define FC2BP_DW_ADDR (3787*1024)
#define FC1BP_DW_ADDR (3790*1024)
#define CONV3BP_DW_ADDR (5413*1024)
#define CONV2BP_DW_ADDR (5450*1024)
#define CONV1BP_DW_ADDR (5483*1024)

#define RMEM_ADDR0 0 
#define RMEM_ADDR1 12800
#define RMEM_ADDR2 17984 
#define RMEM_ADDR3 21120 
#define RMEM_ADDR4 21632 

static const uint32_t drlp_coord_x = DRLP_X;
static const uint32_t drlp_coord_y = DRLP_Y;
static const uint32_t cfg_addr[DRLP_CFG_LEN] = DRLP_CFG_ADDR;
static const uint32_t dram_coord_x = DRAM_X;
static const uint32_t dram_coord_y = DRAM_Y;

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

uint32_t read_rmem(hb_mc_manycore_t *mc, uint32_t rmem_r_addr) {
	/******************************/
	/* Read back config from DRLP */
	/******************************/
	int err;
	uint32_t rmem_r_data;
	uint32_t mc_rmem_r_addr = (DRLP_RMEM_PREFIX+rmem_r_addr)*4;

	hb_mc_npa_t npa = { .x = drlp_coord_x, .y = drlp_coord_y, .epa = mc_rmem_r_addr };
	err = hb_mc_manycore_read_mem(mc, &npa, &rmem_r_data, sizeof(rmem_r_data));
	if (err != HB_MC_SUCCESS) {
		bsg_pr_err("%s: failed to read from manycore DRLP's RMEM: %s\n", __func__, hb_mc_strerror(err));
		hb_mc_manycore_exit(mc);
		return err;
	}

	return rmem_r_data;
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
		if (i % 10000 == 1)
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

void read_dram (hb_mc_manycore_t *mc, uint32_t base_addr, int len) {
	uint32_t read_data;
	int err;
	for (size_t i = 0; i < len; i++) {
		hb_mc_npa_t npa = { .x = dram_coord_x, .y = dram_coord_y , .epa = base_addr*4 + (i*4) };
		err = hb_mc_manycore_read_mem(mc, &npa,
					      &read_data, sizeof(read_data));
		if (err != HB_MC_SUCCESS) {
			bsg_pr_err("%s: failed to read A[%d] "
				   "from DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
				   i, dram_coord_x, dram_coord_y, base_addr + i);
		}
		bsg_pr_test_info("Read result(%d) %x \n", i, read_data);
	}
}

void dram2file (hb_mc_manycore_t *mc, FILE *f, uint32_t base_addr, int len) {
	uint32_t read_data;
	int err;
	base_addr = base_addr + 4;
	for (size_t i = 0; i < len; i++) {
		hb_mc_npa_t npa = { .x = dram_coord_x, .y = dram_coord_y , .epa = base_addr*4 + (i*4) };
		err = hb_mc_manycore_read_mem(mc, &npa,
					      &read_data, sizeof(read_data));
		if (err != HB_MC_SUCCESS) {
			bsg_pr_err("%s: failed to read A[%d] "
				   "from DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
				   i, dram_coord_x, dram_coord_y, base_addr + i);
		}
		fprintf(f, "%08x\n", read_data);
	}
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
	uint32_t config[DRLP_CFG_LEN] = {0x84C0011C, RMEM_ADDR3, FC2_WGT_ADDR, RMEM_ADDR4, 0, 0x02B80028, 0x4400009B};
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

int fc2_dw (hb_mc_manycore_t *mc) {
	int err = HB_MC_SUCCESS;
	uint32_t config[DRLP_CFG_LEN] = {0xC7800203, OUT_GD_ADDR, RMEM_ADDR3+1, FC2BP_DW_ADDR, 270, 0x0020002c, 0x5300089B}; // 270 could be any numbers
	uint32_t done = 0;
	err = write_configure(mc, config);
	// Wait for stop
	hb_mc_npa_t done_npa = { .x = drlp_coord_x, .y = drlp_coord_y, .epa = DRLP_DONE_ADDR };
	while (done != 1) {
		for (int i=0; i<999; i++){}
		hb_mc_manycore_read_mem(mc, &done_npa, &done, sizeof(done));
	}
	bsg_pr_test_info("FC2_dw DONE\n");
	return err;
}

int fc2_dx (hb_mc_manycore_t *mc) {
	int err = HB_MC_SUCCESS;
	uint32_t config[DRLP_CFG_LEN] = {0x87C02000, OUT_GD_ADDR, FC2BP_WGT_ADDR, RMEM_ADDR4+1, RMEM_ADDR3+1, 0x00180028, 0x66004C03};
	uint32_t done = 0;
	err = write_configure(mc, config);
	// Wait for stop
	hb_mc_npa_t done_npa = { .x = drlp_coord_x, .y = drlp_coord_y, .epa = DRLP_DONE_ADDR };
	while (done != 1) {
		for (int i=0; i<999; i++){}
		hb_mc_manycore_read_mem(mc, &done_npa, &done, sizeof(done));
	}
	bsg_pr_test_info("FC2_dx DONE\n");
	return err;
}

int fc1_dw (hb_mc_manycore_t *mc) {
	int err = HB_MC_SUCCESS;
	uint32_t config[DRLP_CFG_LEN] = {0xC7C10BFF, RMEM_ADDR4+1, RMEM_ADDR2+1, FC1BP_DW_ADDR, 288, 0x1000002C, 0x7100FFFF}; // 288 could be any numbers
	uint32_t done = 0;
	err = write_configure(mc, config);
	// Wait for stop
	hb_mc_npa_t done_npa = { .x = drlp_coord_x, .y = drlp_coord_y, .epa = DRLP_DONE_ADDR };
	while (done != 1) {
		for (int i=0; i<999; i++){}
		hb_mc_manycore_read_mem(mc, &done_npa, &done, sizeof(done));
	}
	bsg_pr_test_info("FC1_dw DONE\n");
	return err;
}

int fc1_dx (hb_mc_manycore_t *mc) {
	int err = HB_MC_SUCCESS;
	uint32_t config[DRLP_CFG_LEN] = {0x87C0201C, RMEM_ADDR4+1, FC1BP_WGT_ADDR, RMEM_ADDR3+1, RMEM_ADDR2+1, 0x02B80028, 0x84004C03};
	uint32_t done = 0;
	for (int k = 0; k < 7; ++k){
		err = write_configure(mc, config);
		// Wait for stop
		hb_mc_npa_t done_npa = { .x = drlp_coord_x, .y = drlp_coord_y, .epa = DRLP_DONE_ADDR };
		while (done != 1) {
			for (int i=0; i<999; i++){}
			hb_mc_manycore_read_mem(mc, &done_npa, &done, sizeof(done));
		}
		bsg_pr_test_info("FC1_dx %d/7 DONE\n",k+1);
		done = 0;
		if (k == 0) {
			config[2] = config[2] + 267804; //267805 or 267804?
			config[5] = 0x02b80008;
		} else {
			config[2] = config[2] + 267804;
		}
		config[3] = config[3] + 8; // was 512
		config[4] = config[4] + 8; // was 488
	}
	return err;
}

int conv3_dw (hb_mc_manycore_t *mc) {
	int err = HB_MC_SUCCESS;
	uint32_t config[DRLP_CFG_LEN] = {0x0FFF0308, RMEM_ADDR1+1, RMEM_ADDR3+1, CONV3BP_DW_ADDR, 9216, 0xA2000028, 0x9100FFFF};
	uint32_t done = 0;
	for (int k = 0; k < 4; ++k){
		err = write_configure(mc, config);
		// Wait for stop
		hb_mc_npa_t done_npa = { .x = drlp_coord_x, .y = drlp_coord_y, .epa = DRLP_DONE_ADDR };
		while (done != 1) {
			for (int i=0; i<999; i++){}
			hb_mc_manycore_read_mem(mc, &done_npa, &done, sizeof(done));
		}
		bsg_pr_test_info("CONV3_dw %d/4 DONE\n", k+1);
		done = 0;
		config[2] = config[2] + 896; //was 16
		config[3] = config[3] + 9216;
		config[5] = 0xA2000008;
	}
	return err;
}

int conv3_dx (hb_mc_manycore_t *mc) {
	int err = HB_MC_SUCCESS;
	uint32_t config[DRLP_CFG_LEN] = {0x07C82008, RMEM_ADDR3+1, CONV3BP_WGT_ADDR, RMEM_ADDR2+1, RMEM_ADDR1+1, 0x63000028, 0xA400FFFF};
	uint32_t done = 0;
	for (int k = 0; k < 4; ++k){
		err = write_configure(mc, config);
		// Wait for stop
		hb_mc_npa_t done_npa = { .x = drlp_coord_x, .y = drlp_coord_y, .epa = DRLP_DONE_ADDR };
		while (done != 1) {
			for (int i=0; i<999; i++){}
			hb_mc_manycore_read_mem(mc, &done_npa, &done, sizeof(done));
		}
		bsg_pr_test_info("CONV3_dX %d/4 DONE\n", k+1);
		done = 0;
		config[2] = config[2] + 9232;
		config[3] = config[3] + 1296;
		config[4] = config[4] + 1296; // was config[4] = config[3] + 1296 before
		config[5] = 0x63000008;
	}
	return err;
}

int conv2_dw (hb_mc_manycore_t *mc) {
	int err = HB_MC_SUCCESS;
	uint32_t config[DRLP_CFG_LEN] = {0x0FDF050F, RMEM_ADDR0+1, RMEM_ADDR2+1, CONV2BP_DW_ADDR, 8192, 0xF0000028, 0xB100FFFF};
	uint32_t done = 0;
	for (int k = 0; k < 4; ++k){
		err = write_configure(mc, config);
		// Wait for stop
		hb_mc_npa_t done_npa = { .x = drlp_coord_x, .y = drlp_coord_y, .epa = DRLP_DONE_ADDR };
		while (done != 1) {
			for (int i=0; i<999; i++){}
			hb_mc_manycore_read_mem(mc, &done_npa, &done, sizeof(done));
		}
		bsg_pr_test_info("CONV2_dW %d/4 DONE\n", k+1);
		done = 0;
		config[2] = config[2] + 16*81;
		config[3] = config[3] + 8192;
		config[5] = 0x01000008;
	}
	return err;
}

int conv2_dx (hb_mc_manycore_t *mc) {
	int err = HB_MC_SUCCESS;
	uint32_t config[DRLP_CFG_LEN] = {0x4BC91009, RMEM_ADDR2+1, CONV2BP_WGT_ADDR, RMEM_ADDR1+1, RMEM_ADDR0+1, 0x6E000028, 0xC400FFFF};
	uint32_t done = 0;
	for (int k = 0; k < 8; ++k){
		err = write_configure(mc, config);
		// Wait for stop
		hb_mc_npa_t done_npa = { .x = drlp_coord_x, .y = drlp_coord_y, .epa = DRLP_DONE_ADDR };
		while (done != 1) {
			for (int i=0; i<999; i++){}
			hb_mc_manycore_read_mem(mc, &done_npa, &done, sizeof(done));
		}
		bsg_pr_test_info("CONV2_dX %d/8 DONE\n", k+1);
		done = 0;
		if (k==0 || k==2 || k==4 || k==6) {
			config[3] = config[3] + 6400;
			config[4] = config[4] + 6400;
		}
		else if (k==1 || k==5) {
			config[3] = config[3] - 6400 + 20;
			config[4] = config[4] - 6400 + 20;
		}
		else {
			config[3] = config[3] - 6400 - 19;
			config[4] = config[4] - 6400 - 19;
		}
		config[2] = config[2] + 4112;
		config[5] = 0x63000008;
	}
	return err;
}

int conv1_dw (hb_mc_manycore_t *mc) {
	int err = HB_MC_SUCCESS;
	uint32_t config[DRLP_CFG_LEN] = {0x0FC0173F, CONV1BP_ACT_ADDR, RMEM_ADDR1+1, CONV1BP_DW_ADDR, 1024, 0x8A000028, 0xD300FFFF};
	uint32_t done = 0;
	for (int k = 0; k < 8; ++k){
		err = write_configure(mc, config);
		// Wait for stop
		hb_mc_npa_t done_npa = { .x = drlp_coord_x, .y = drlp_coord_y, .epa = DRLP_DONE_ADDR };
		while (done != 1) {
			for (int i=0; i<999; i++){}
			hb_mc_manycore_read_mem(mc, &done_npa, &done, sizeof(done));
		}
		bsg_pr_test_info("CONV1_dW %d/8 DONE\n", k+1);
		done = 0;
		if (k==1 || k==3 || k==5) {
			config[1] = config[1] + 26496;
			config[2] = config[2] - 16*400;
			config[5] = 0x8A000028;
		}
		else {
			config[2] = config[2] + 16*400;
			config[5] = 0x8A000008;
		}
		config[3] = config[3] + 1024;
	}
	return err;
}
