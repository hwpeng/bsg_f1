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

#include "test_drlp_dqn.h"
#define TEST_NAME "test_drlp_dqn"
#define ALLOC_NAME "default_allocator"

int cuda_optimizer (hb_mc_device_t device, char *bin_path, eva_t w_device, eva_t dw_device, eva_t w_new_device, float *w, float *dw, int w_num, float lr) {
	int rc;
    rc = hb_mc_device_program_init(&device, bin_path, ALLOC_NAME, 0);
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to initialize program.\n");
            return rc;
    }
    else {
        bsg_pr_test_info("Initialize program %s. \n", bin_path);
    }

    /* Copy W & dW from host onto device DRAM (eva) */
	bsg_pr_test_info("========Copy from host to device DRAM (eva)========\n");
    void *dst = (void *) ((intptr_t) w_device);
    void *src = (void *) &w[0];
    rc = hb_mc_device_memcpy (&device, dst, src, w_num * sizeof(uint32_t), HB_MC_MEMCPY_TO_DEVICE); 
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to copy memory to device.\n");
            return rc;
    }
    dst = (void *) ((intptr_t) dw_device);
    src = (void *) &dw[0];
    rc = hb_mc_device_memcpy (&device, dst, src, w_num * sizeof(uint32_t), HB_MC_MEMCPY_TO_DEVICE); 
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to copy memory to device.\n");
            return rc;
    }

	/* Initialize values in w_new_device to 0. */
    rc = hb_mc_device_memset(&device, &w_new_device, 0, w_num * sizeof(uint32_t));
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to set memory on device.\n");
            return rc;
    } 

    /* Define block_size_x/y: amount of work for each tile group */
    /* Define tg_dim_x/y: number of tiles in each tile group */
    /* Calculate grid_dim_x/y: number of tile groups needed based on block_size_x/y */
    uint32_t block_size_x = w_num;
    hb_mc_dimension_t tg_dim = { .x = 2, .y = 2}; 
    hb_mc_dimension_t grid_dim = { .x = 1, .y = 1}; 

    /* Prepare list of input arguments for kernel. */
    int cuda_argv[6] = {w_device, dw_device, w_new_device, lr, w_num, block_size_x};

    /* Enquque grid of tile groups, pass in grid and tile group dimensions, kernel name, number and list of input arguments */
	bsg_pr_test_info("========Enqueue cuda kernel========\n");
    rc = hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_optimizer", 6, cuda_argv);
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to initialize grid.\n");
            return rc;
    }

    /* Launch and execute all tile groups on device and wait for all to finish.  */
	bsg_pr_test_info("========Excute cuda kernel========\n");
    rc = hb_mc_device_tile_groups_execute(&device);
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to execute tile groups.\n");
            return rc;
    }

    /* Copy result matrix back from device DRAM into host memory. */
	bsg_pr_test_info("========Copy from device DRAM to host========\n");
    src = (void *) ((intptr_t) w_new_device);
    dst = (void *) &w[0];
    rc = hb_mc_device_memcpy (&device, (void *) dst, src, w_num * sizeof(uint32_t), HB_MC_MEMCPY_TO_HOST);
    if (rc != HB_MC_SUCCESS) { 
		bsg_pr_err("failed to copy memory from device.\n");
		return rc;
    }
}


int test_drlp_dqn (int argc, char **argv) {

	bsg_pr_test_info("Running DRLP-CUDA DQN test!\n");

    /*****************************************************************************************************************
	* Test game and python settings
    ******************************************************************************************************************/
	char *game_name="CartPole-v1";
	PyObject *pinst;
	pinst = py_init(game_name); // Initialize python class instance and method

    /*****************************************************************************************************************
    * NN configuration 
    ******************************************************************************************************************/
	int num_layer = 2;
	FC_layer FC1 = {.input_size=STATE_SIZE, 
					.output_size=FC1_Y_SIZE,
					.weight_size=FC1_W_SIZE,
					.input_src=DMA,
					.output_dst=ONCHIP,
					.relu=1,
					.layer=3,
					.act_base_addr=STATE_ADDR,
					.wgt_base_addr=FC1_W_ADDR,
					.rst_base_addr=FC1_Y_ADDR,
					.dy_base_addr=FC2_dX_ADDR,
					.dw_base_addr=FC1_dW_ADDR
	};
	FC_layer FC2 = {.input_size=FC1_Y_SIZE, 
					.output_size=FC2_Y_SIZE,
					.weight_size=FC2_W_SIZE,
					.input_src=ONCHIP,
					.output_dst=DMA,
					.relu=0,
					.layer=4,
					.act_base_addr=FC1_Y_ADDR,
					.wgt_base_addr=FC2_W_ADDR,
					.rst_base_addr=FC2_Y_ADDR,
					.dy_base_addr=FC2_dY_ADDR,
					.dw_base_addr=FC2_dW_ADDR,
					.wT_base_addr=FC2_WT_ADDR,
					.dx_base_addr=FC2_dX_ADDR
	};
	FC_layer nn[2] = {FC1, FC2};

    /*****************************************************************************************************************
    * Initialize device 
    ******************************************************************************************************************/
	int rc;
	char *bin_path;
	/* bin_path = "/mnt/users/ssd1/homes/huwan/bsg/bsg_bladerunner/bsg_manycore/software/spmd/bsg_cuda_lite_runtime/drlp_cuda/main.riscv"; */
	bin_path = "../../../bsg_manycore/software/spmd/bsg_cuda_lite_runtime/drlp_cuda/main.riscv";
    hb_mc_device_t device;
    rc = hb_mc_device_init(&device, TEST_NAME, 0);
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to initialize device.\n");
            return rc;
    }
	hb_mc_manycore_t *mc = device.mc;

    /*****************************************************************************************************************
    * CUDA optimizer 
    ******************************************************************************************************************/
    rc = hb_mc_device_program_init(&device, bin_path, ALLOC_NAME, 0);
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to initialize program.\n");
            return rc;
    }
	else {
		bsg_pr_test_info("Initialize program %s. \n", bin_path);
	}

    /* Allocate memory on the device for W, dW, and W_NEW */
	bsg_pr_test_info("========Allocate memory on device========\n");
    eva_t w1_opt_eva, dw1_opt_eva, w1_new_opt_eva; 
    eva_t b1_opt_eva, db1_opt_eva, b1_new_opt_eva; 
    eva_t w2_opt_eva, dw2_opt_eva, w2_new_opt_eva; 
    eva_t b2_opt_eva, db2_opt_eva, b2_new_opt_eva; 
    hb_mc_device_malloc(&device, FC1_W_SIZE * sizeof(uint32_t), &w1_opt_eva); 
    hb_mc_device_malloc(&device, FC1_W_SIZE * sizeof(uint32_t), &dw1_opt_eva); 
    hb_mc_device_malloc(&device, FC1_W_SIZE * sizeof(uint32_t), &w1_new_opt_eva); 
    hb_mc_device_malloc(&device, FC1_Y_SIZE * sizeof(uint32_t), &b1_opt_eva); 
    hb_mc_device_malloc(&device, FC1_Y_SIZE * sizeof(uint32_t), &db1_opt_eva); 
    hb_mc_device_malloc(&device, FC1_Y_SIZE * sizeof(uint32_t), &b1_new_opt_eva); 
    hb_mc_device_malloc(&device, FC2_W_SIZE * sizeof(uint32_t), &w2_opt_eva); 
    hb_mc_device_malloc(&device, FC2_W_SIZE * sizeof(uint32_t), &dw2_opt_eva); 
    hb_mc_device_malloc(&device, FC2_W_SIZE * sizeof(uint32_t), &w2_new_opt_eva); 
    hb_mc_device_malloc(&device, FC2_Y_SIZE * sizeof(uint32_t), &b2_opt_eva); 
    hb_mc_device_malloc(&device, FC2_Y_SIZE * sizeof(uint32_t), &db2_opt_eva); 
    hb_mc_device_malloc(&device, FC2_Y_SIZE * sizeof(uint32_t), &b2_new_opt_eva); 

    /* Allocate memory on the device for replay memory*/
    hb_mc_coordinate_t target = { .x = 1, .y = 1 };
    size_t re_mem_size = sizeof(uint32_t)*TRANSITION_SIZE*RE_MEM_SIZE;
    eva_t re_mem_eva;
    hb_mc_npa_t re_mem_npa;
    hb_mc_idx_t re_mem_x, re_mem_y;
    hb_mc_epa_t re_mem_epa;

    hb_mc_device_malloc(&device, re_mem_size, &re_mem_eva); 
    hb_mc_eva_to_npa(mc, &default_map, &target, &re_mem_eva, &re_mem_npa, &re_mem_size);
    re_mem_x = hb_mc_npa_get_x(&re_mem_npa);
    re_mem_y = hb_mc_npa_get_y(&re_mem_npa);
    re_mem_epa = hb_mc_npa_get_epa(&re_mem_npa);
    
    bsg_pr_test_info("Replay memory: EVA 0x%x mapped to NPA (x: %d, y: %d, EPA, 0x%x)\n", hb_mc_eva_addr(&re_mem_eva), re_mem_x, re_mem_y, re_mem_epa);


    /*****************************************************************************************************************
    * Weight random initialization and write to dram
    ******************************************************************************************************************/
	// On the host
	float FC1_W[FC1_W_SIZE];
	float FC1_B[FC1_Y_SIZE];
	float FC2_W[FC2_W_SIZE];
	float FC2_B[FC2_Y_SIZE];
	srand(0.1); 
	param_random(FC1_W, FC1_W_SIZE);
	param_random(FC1_B, FC1_Y_SIZE);
	param_random(FC2_W, FC2_W_SIZE);
	param_random(FC2_B, FC2_Y_SIZE);
	// To the device DRAM
	uint32_t base_addr = FC1_W_ADDR;
	fc_fp_wrt_wgt(mc, FC1, FC1_W, FC1_B, base_addr);
	base_addr = FC2_W_ADDR;
	fc_fp_wrt_wgt(mc, FC2, FC2_W, FC2_B, base_addr);

	float FC2_WT[FC2_W_SIZE];
    /*****************************************************************************************************************
    * DQN  
    ******************************************************************************************************************/
	// Replay memory init
	uint32_t position = 0;
	Transition trans;
	call_reset(&trans, pinst);
	for (int i = 0; i < RE_MEM_INIT_SIZE; i++) {
		trans.action = rand() % ACTION_SIZE;
		call_step(&trans, pinst);
		position = re_mem_push(mc, re_mem_npa, &trans, position);
		if (trans.done==0) {
			for (int j=0; j<STATE_SIZE; j++)
				trans.state[j] = trans.next_state[j];
		}
		else {
			call_reset(&trans, pinst);
		}
	}

	/* read_re_mem(mc, 0, 40);  */
	
	// for (int i = 0; i < RE_MEM_INIT_SIZE; i++) {
	// 	rc = re_mem_sample(mc, &trans, RE_MEM_INIT_SIZE);
	// 	for (int j=0; j<STATE_SIZE; j++) {
	// 		printf("State[%d]=%1.4f\t", j, trans.state[j]);
	// 	}
	// 	printf("\n");
	// 	for (int j=0; j<STATE_SIZE; j++) {
	// 		printf("NextState[%d]=%1.4f\t", j, trans.next_state[j]);
	// 	}
	// 	printf("\n");
	// 	printf("Action=%d\n", trans.action);
	// 	printf("Reward=%1.1f\n", trans.reward);
	// 	printf("Done=%d\n", trans.done);
	// }

	// Training loop 
	float epsilon=0.0;
	int num_trans;
	bool re_mem_full = false;
	bool compare_host = true;
	Transition sample_trans;
	float FC1_dW[FC1_W_SIZE], FC1_dB[FC1_Y_SIZE] = {0.0};
	float FC2_dW[FC2_W_SIZE], FC2_dB[FC2_Y_SIZE] = {0.0};
	float host_fc2_w_new[FC2_W_SIZE];
	float host_fc1_w_new[FC1_W_SIZE];
	for (int step = 0; step < STEP_MAX; step++) {
		bsg_pr_test_info("Step%d\n", step);

		// Perform one step
		bsg_pr_test_info("Perform one step\n");
		dqn_act(mc, &trans, nn, num_layer, epsilon);
		call_step(&trans, pinst);

		// Push to replay memory 
		bsg_pr_test_info("Push to replay memory\n");
		position = re_mem_push(mc, re_mem_npa, &trans, position);
		if (position == 0)
			re_mem_full = true;
		if (re_mem_full)
			num_trans = RE_MEM_SIZE;
		else
			num_trans = position;

		if (trans.done==0) {
			for (int j=0; j<STATE_SIZE; j++)
				trans.state[j] = trans.next_state[j];
		}
		else {
			call_reset(&trans, pinst);
		}

		// Training 
		if (step%TRAIN_FREQ==0) {
			// Weight transpose and write
			bsg_pr_test_info("Weight transpose and write\n");
			wgt_transpose_and_write(mc, FC2, FC2_W, FC2_WT);

			// Sample from replay memory
			bsg_pr_test_info("Sample from replay memory\n");
			re_mem_sample(mc, re_mem_npa, &sample_trans, num_trans);

			// Train
			dqn_train(mc, &sample_trans, nn, num_layer, FC2_dB, 0.95);
			read_dw(mc, FC2_dW, FC2);
			read_dw(mc, FC1_dW, FC1);
			read_db(mc, FC1_dB, FC1);
			if (HOST_COMPARE) {
				rc = host_train(sample_trans.state, sample_trans.next_state, sample_trans.reward, sample_trans.done, FC1_W, FC1_B, FC2_W, FC2_B, FC2_WT, FC2_dW, FC1_dW, STATE_SIZE, FC1_Y_SIZE, ACTION_SIZE); 
				/* if (rc==1) */
					/* bsg_pr_err("Step%d, BP has error!\n", step); */
				host_optimizer(host_fc2_w_new, FC2_W, FC2_dW, LR, FC2_W_SIZE);
				host_optimizer(host_fc1_w_new, FC1_W, FC1_dW, LR, FC1_W_SIZE);
			}

			// Optimizer
			// cuda_optimizer(device, bin_path, w2_opt_eva, dw2_opt_eva, w2_new_opt_eva, FC2_W, FC2_dW, FC2_W_SIZE, LR);
			// cuda_optimizer(device, bin_path, w1_opt_eva, dw1_opt_eva, w1_new_opt_eva, FC1_W, FC1_dW, FC1_W_SIZE, LR);
			// cuda_optimizer(device, bin_path, b2_opt_eva, db2_opt_eva, b2_new_opt_eva, FC2_B, FC2_dB, FC2_Y_SIZE, LR);
			// cuda_optimizer(device, bin_path, b1_opt_eva, db1_opt_eva, b1_new_opt_eva, FC1_B, FC1_dB, FC1_Y_SIZE, LR);
			// /* for (int i = 0; i < FC1_Y_SIZE; i++)  */
            //     /* bsg_pr_test_info("F1_B[%d]=%f\n", i, FC1_B[i]); */

			// if (HOST_COMPARE) {
			// 	rc = host_compare(host_fc2_w_new, FC2_W, FC2_W_SIZE);
			// 	rc = host_compare(host_fc1_w_new, FC1_W, FC1_W_SIZE);
			// 	if (rc==1)
			// 		bsg_pr_err("Step%d, optimizer has error!\n", step);
			// }
			// // Write new weight to DRAM
			// base_addr = FC1_W_ADDR;
			// fc_fp_wrt_wgt(mc, FC1, FC1_W, FC1_B, base_addr);
			// base_addr = FC2_W_ADDR;
			// fc_fp_wrt_wgt(mc, FC2, FC2_W, FC2_B, base_addr);

		}
	}
    float read_float[2000];
    read_dram(mc, 46199, 2000, read_float, true);	

    /* Freeze the tiles and memory manager cleanup. */
	Py_DECREF(pinst);	
    Py_Finalize(); 
	printf("=========================\n");
    rc = hb_mc_device_finish(&device); 
    if (rc != HB_MC_SUCCESS) { 
		bsg_pr_err("failed to de-initialize device.\n");
    	return rc;
    }


	return HB_MC_SUCCESS;
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
	int rc = test_drlp_dqn(argc, argv);
	*exit_code = rc;
	bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
	return;
}
#else
int main(int argc, char ** argv) {
	bsg_pr_test_info(TEST_NAME " Regression Test (F1)\n");
	int rc = test_drlp_dqn(argc, argv);
	bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
	return rc;
}
#endif
