static const uint32_t cfg_addr[DRLP_CFG_LEN] = DRLP_CFG_ADDR;

int write_configure(hb_mc_manycore_t *mc, uint32_t config_array[DRLP_CFG_LEN]) {
	uint32_t config;
	int err;
	for (size_t i = 0; i < DRLP_CFG_LEN; i++) {
		config = config_array[i];
		hb_mc_npa_t npa = { .x = DRLP_X, .y = DRLP_Y, .epa = cfg_addr[i] };
		err = hb_mc_manycore_write_mem(mc, &npa, &config, sizeof(config));
		if (err != HB_MC_SUCCESS) {
			bsg_pr_err("%s: failed to write to DRLP configure registers: %s\n", __func__, hb_mc_strerror(err));
			hb_mc_manycore_exit(mc);
			return err;
		}
	}
	// bsg_pr_test_info("Write configure successful\n");
	// Turn off drlp
	config = config-1;
	hb_mc_npa_t npa = { .x = DRLP_X, .y = DRLP_Y, .epa = cfg_addr[DRLP_CFG_LEN-1] };
	err = hb_mc_manycore_write_mem(mc, &npa, &config, sizeof(config));
	if (err != HB_MC_SUCCESS) {
		bsg_pr_err("%s: failed to write to DRLP configure registers: %s\n", __func__, hb_mc_strerror(err));
		hb_mc_manycore_exit(mc);
	}
	return err;
}


int fc2_dw (hb_mc_manycore_t *mc) {
	int err = HB_MC_SUCCESS;
	uint32_t config[DRLP_CFG_LEN] = {0xC7800203, OUT_GD_ADDR, RMEM_ADDR3+1, FC2BP_DW_ADDR, 270, 0x0020002c, 0x5300089B}; // 270 could be any numbers
	uint32_t done = 0;
	err = write_configure(mc, config);
	// Wait for stop
	hb_mc_npa_t done_npa = { .x = DRLP_X, .y = DRLP_Y, .epa = DRLP_DONE_ADDR };
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
	hb_mc_npa_t done_npa = { .x = DRLP_X, .y = DRLP_Y, .epa = DRLP_DONE_ADDR };
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
	hb_mc_npa_t done_npa = { .x = DRLP_X, .y = DRLP_Y, .epa = DRLP_DONE_ADDR };
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
	hb_mc_npa_t done_npa = { .x = DRLP_X, .y = DRLP_Y, .epa = DRLP_DONE_ADDR };
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
	    hb_mc_npa_t done_npa = { .x = DRLP_X, .y = DRLP_Y, .epa = DRLP_DONE_ADDR };
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
	    hb_mc_npa_t done_npa = { .x = DRLP_X, .y = DRLP_Y, .epa = DRLP_DONE_ADDR };
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
	    hb_mc_npa_t done_npa = { .x = DRLP_X, .y = DRLP_Y, .epa = DRLP_DONE_ADDR };
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
	    hb_mc_npa_t done_npa = { .x = DRLP_X, .y = DRLP_Y, .epa = DRLP_DONE_ADDR };
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
	    hb_mc_npa_t done_npa = { .x = DRLP_X, .y = DRLP_Y, .epa = DRLP_DONE_ADDR };
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


