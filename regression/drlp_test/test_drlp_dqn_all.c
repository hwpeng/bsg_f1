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

#include "test_drlp_dqn_all.h"
#define TEST_NAME "test_drlp_dqn_all"
#define ALLOC_NAME "default_allocator"

int cuda_optimizer (hb_mc_device_t device, char *bin_path, eva_t w_device, eva_t dw_device, eva_t w_new_device, float *w, float *dw, int w_num, float lr) {
	int rc;
	/* bsg_pr_test_info("========Copy from host to device DRAM (eva)========\n"); */
    void *dst = (void *) ((intptr_t) w_device);
    void *src = (void *) &w[0];
    rc = hb_mc_device_memcpy (&device, dst, src, w_num * sizeof(uint32_t), HB_MC_MEMCPY_TO_DEVICE); 
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to copy memory to device.\n");
            return rc;
    }
    /* bsg_pr_test_info("w[0]=%.16f \n", w[0]); */

	hb_mc_manycore_t *mc = device.mc;
    hb_mc_coordinate_t tt = { .x = 1, .y = 1 };
    size_t w1_size = sizeof(uint32_t)*FC1_W_SIZE;
    hb_mc_npa_t w1_npa, w1_new_npa;
    hb_mc_idx_t w1_x, w1_y, w1_new_x, w1_new_y;
    hb_mc_epa_t w1_epa, w1_new_epa;

    hb_mc_eva_to_npa(mc, &default_map, &tt, &w_device, &w1_npa, &w1_size);
    w1_x = hb_mc_npa_get_x(&w1_npa);
    w1_y = hb_mc_npa_get_y(&w1_npa);
    w1_epa = hb_mc_npa_get_epa(&w1_npa);
    /* bsg_pr_test_info("w1 opt: EVA 0x%x mapped to NPA (x: %d, y: %d, EPA, %d)\n", hb_mc_eva_addr(&w_device), w1_x, w1_y, w1_epa); */

    hb_mc_npa_t npa = { .x = w1_x, .y = w1_y, .epa = w1_epa };
	uint32_t read_data;
    float read_float;
    hb_mc_manycore_read_mem(mc, &npa, &read_data, sizeof(read_data));
    read_float = flt(read_data);
    /* bsg_pr_test_info("Read result %1.4f(%x) \n", read_float, read_data); */

    dst = (void *) ((intptr_t) dw_device);
    src = (void *) &dw[0];
    rc = hb_mc_device_memcpy (&device, dst, src, w_num * sizeof(uint32_t), HB_MC_MEMCPY_TO_DEVICE); 
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to copy memory to device.\n");
            return rc;
    }
    /* bsg_pr_test_info("dw[0]=%.16f \n", dw[0]); */

	/* Initialize values in w_new_device to 0. */
    rc = hb_mc_device_memset(&device, &w_new_device, 0, w_num * sizeof(uint32_t));
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to set memory on device.\n");
            return rc;
    } 
    hb_mc_manycore_read_mem(mc, &npa, &read_data, sizeof(read_data));
    read_float = flt(read_data);
    /* bsg_pr_test_info("Read result %1.4f(%x) \n", read_float, read_data); */

    /* Define block_size_x/y: amount of work for each tile group */
    /* Define tg_dim_x/y: number of tiles in each tile group */
    /* Calculate grid_dim_x/y: number of tile groups needed based on block_size_x/y */
    uint32_t block_size_x = w_num;
    hb_mc_dimension_t tg_dim = { .x = 2, .y = 2}; 
    hb_mc_dimension_t grid_dim = { .x = 1, .y = 1}; 

    /* Prepare list of input arguments for kernel. */
    int cuda_argv[6] = {w_device, dw_device, w_new_device, lr, w_num, block_size_x};

    /* Enquque grid of tile groups, pass in grid and tile group dimensions, kernel name, number and list of input arguments */
	/* bsg_pr_test_info("========Enqueue cuda kernel========\n"); */
    rc = hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_optimizer", 6, cuda_argv);
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to initialize grid.\n");
            return rc;
    }

    /* Launch and execute all tile groups on device and wait for all to finish.  */
	/* bsg_pr_test_info("========Excute cuda kernel========\n"); */
    rc = hb_mc_device_tile_groups_execute(&device);
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to execute tile groups.\n");
            return rc;
    }

    /* Copy result matrix back from device DRAM into host memory. */
	/* bsg_pr_test_info("========Copy from device DRAM to host========\n"); */
    src = (void *) ((intptr_t) w_new_device);
    dst = (void *) &w[0];
    rc = hb_mc_device_memcpy (&device, (void *) dst, src, w_num * sizeof(uint32_t), HB_MC_MEMCPY_TO_HOST);
    if (rc != HB_MC_SUCCESS) { 
		bsg_pr_err("failed to copy memory from device.\n");
		return rc;
    }
}


int test_drlp_dqn_all (int argc, char **argv) {

    bsg_pr_test_info("Running DRLP-CUDA DQN Breakout test!\n");

    /*****************************************************************************************************************
    * Test game and python settings
    ******************************************************************************************************************/
    char *game_name="Breakout-v0";
    PyObject *pinst;
    pinst = py_init(game_name); // Initialize python class instance and method

    /*****************************************************************************************************************
    * NN configuration 
    ******************************************************************************************************************/
    int num_layer = 5;
    NN_layer CONV1 = {.input_size=STATE_SIZE, 
                      .output_size=CONV1_Y_SIZE,
                      .weight_size=CONV1_W_SIZE,
                      .bias_size=32,
                      .input_src=DMA,
                      .output_dst=ONCHIP,
                      .relu=1,
                      .layer=0,
                      .FC_CONV=1,
                      .act_base_addr=STATE_ADDR,
                      .wgt_base_addr=CONV1_WGT_ADDR,
                      .rst_base_addr=RMEM_ADDR0,
                      .dy_base_addr=RMEM_ADDR1+1,
                      .wT_base_addr=CONV1BP_ACT_ADDR, // just for conv1
                      .dw_base_addr=CONV1BP_DW_ADDR
    };
    NN_layer CONV2 = {.input_size=CONV1_Y_SIZE, 
                      .output_size=CONV2_Y_SIZE,
                      .weight_size=CONV2_W_SIZE,
                      .bias_size=64,
                      .input_src=ONCHIP,
                      .output_dst=ONCHIP,
                      .relu=1,
                      .layer=1,
                      .FC_CONV=1,
                      .act_base_addr=RMEM_ADDR0,
                      .wgt_base_addr=CONV2_WGT_ADDR,
                      .rst_base_addr=RMEM_ADDR1,
                      .dy_base_addr=RMEM_ADDR2+1,
                      .wT_base_addr=CONV2BP_WGT_ADDR,
                      .dw_base_addr=CONV2BP_DW_ADDR,
                      .dx_base_addr=RMEM_ADDR1+1
    };
    NN_layer CONV3 = {.input_size=CONV2_Y_SIZE, 
                      .output_size=CONV3_Y_SIZE,
                      .weight_size=CONV3_W_SIZE,
                      .bias_size=64,
                      .input_src=ONCHIP,
                      .output_dst=ONCHIP,
                      .relu=1,
                      .layer=2,
                      .FC_CONV=1,
                      .act_base_addr=RMEM_ADDR1,
                      .wgt_base_addr=CONV3_WGT_ADDR,
                      .rst_base_addr=RMEM_ADDR2,
                      .dy_base_addr=RMEM_ADDR3+1,
                      .wT_base_addr=CONV3BP_WGT_ADDR,
                      .dw_base_addr=CONV3BP_DW_ADDR,
                      .dx_base_addr=RMEM_ADDR2+1
    };
    NN_layer FC1 = {.input_size=CONV3_Y_SIZE, 
                    .output_size=FC1_Y_SIZE,
                    .weight_size=FC1_W_SIZE,
                    .bias_size=FC1_Y_SIZE,
                    .input_src=ONCHIP,
                    .output_dst=ONCHIP,
                    .relu=1,
                    .layer=3,
                    .FC_CONV=0,
                    .act_base_addr=RMEM_ADDR2,
                    .wgt_base_addr=FC1_WGT_ADDR,
                    .rst_base_addr=RMEM_ADDR3,
                    .dy_base_addr=RMEM_ADDR4+1,
                    .wT_base_addr=FC1BP_WGT_ADDR,
                    .dw_base_addr=FC1BP_DW_ADDR,
                    .dx_base_addr=RMEM_ADDR3+1
    };
    NN_layer FC2 = {.input_size=FC1_Y_SIZE, 
                    .output_size=FC2_Y_SIZE,
                    .weight_size=FC2_W_SIZE,
                    .bias_size=FC2_Y_SIZE,
                    .input_src=ONCHIP,
                    .output_dst=DMA,
                    .relu=0,
                    .layer=4,
                    .FC_CONV=0,
                    .act_base_addr=RMEM_ADDR3,
                    .wgt_base_addr=FC2_WGT_ADDR,
                    .rst_base_addr=FP_RST_ADDR,
                    .dy_base_addr=OUT_GD_ADDR,
                    .dw_base_addr=FC2BP_DW_ADDR,
                    .wT_base_addr=FC2BP_WGT_ADDR,
                    .dx_base_addr=RMEM_ADDR4+1
    };
    NN_layer nn[5] = {CONV1, CONV2, CONV3, FC1, FC2};

    /*****************************************************************************************************************
    * Initialize device 
    ******************************************************************************************************************/
    int rc;
    char *bin_path;
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
    eva_t w_opt_eva[num_layer], dw_opt_eva[num_layer], w_new_opt_eva[num_layer];
    eva_t b_opt_eva[num_layer], db_opt_eva[num_layer], b_new_opt_eva[num_layer];
    optim_malloc(device, nn, num_layer, w_opt_eva, dw_opt_eva, w_new_opt_eva, b_opt_eva, db_opt_eva, b_new_opt_eva);

    /* Allocate memory on the device for DRLP operation*/
    size_t drlp_dram_size = sizeof(uint32_t)*DRLP_DRAM_SIZE;
    hb_mc_eva_t drlp_dram_eva;
    hb_mc_device_malloc(&device, drlp_dram_size, &drlp_dram_eva); 
    write_dram_configure(mc, drlp_dram_eva);

    /* Allocate memory on the device for replay memory*/
    size_t re_mem_size = sizeof(uint32_t)*TRANSITION_SIZE*RE_MEM_SIZE;
    hb_mc_npa_t re_mem_npa;
    malloc_npa(device, re_mem_size, &re_mem_npa);

    /*****************************************************************************************************************
    * Weight random initialization and write to dram
    ******************************************************************************************************************/
    // On the host
    float CONV1_W[CONV1_W_SIZE];
    float CONV1_B[CONV1_B_SIZE] = {0.0};
    float CONV2_W[CONV2_W_SIZE];
    float CONV2_B[CONV2_B_SIZE] = {0.0};
    float CONV3_W[CONV3_W_SIZE];
    float CONV3_B[CONV3_B_SIZE] = {0.0};
    float FC1_W[FC1_W_SIZE];
    float FC1_B[FC1_B_SIZE] = {0.0};
    float FC2_W[FC2_W_SIZE];
    float FC2_B[FC2_B_SIZE] = {0.0};
    float *nn_w[] = {CONV1_W, CONV2_W, CONV3_W, FC1_W, FC2_W};
    float *nn_b[] = {CONV1_B, CONV2_B, CONV3_B, FC1_B, FC2_B};
    bsg_pr_test_info("Initialize weights randomly and write to DRAM\n");
    srand(0.1); 
    param_random(CONV1_W, CONV1_W_SIZE);
    param_random(CONV2_W, CONV2_W_SIZE);
    param_random(CONV3_W, CONV3_W_SIZE);
    param_random(FC1_W, FC1_W_SIZE);
    param_random(FC2_W, FC2_W_SIZE);
    // To the device DRAM
    uint32_t base_addr = CONV1_WGT_ADDR;
    conv_fp_wrt_wgt(mc, drlp_dram_eva, CONV1, CONV1_W, CONV1_B, base_addr);
    base_addr = CONV2_WGT_ADDR;
    conv_fp_wrt_wgt(mc, drlp_dram_eva, CONV2, CONV2_W, CONV2_B, base_addr);
    base_addr = CONV3_WGT_ADDR;
    conv_fp_wrt_wgt(mc, drlp_dram_eva, CONV3, CONV3_W, CONV3_B, base_addr);
    base_addr = FC1_WGT_ADDR;
    fc_fp_wrt_wgt(mc, drlp_dram_eva, FC1, FC1_W, FC1_B, base_addr);
    base_addr = FC2_WGT_ADDR;
    fc_fp_wrt_wgt(mc, drlp_dram_eva, FC2, FC2_W, FC2_B, base_addr);

    bsg_pr_test_info("Write weights to DRAM done!!!\n");

    /*****************************************************************************************************************
    * DQN  
    ******************************************************************************************************************/
    // Replay memory init
    bsg_pr_test_info("Replay memory init\n");
    uint32_t position = 0;
    Transition trans;
    call_reset(&trans, pinst);
    bsg_pr_test_info("Reset done\n");
    for (int i = 0; i < RE_MEM_INIT_SIZE; i++) {
        trans.action = rand() % ACTION_SIZE;
        call_step(&trans, pinst);
        bsg_pr_test_info("call step done\n");
        position = re_mem_push(mc, re_mem_npa, &trans, position);
        bsg_pr_test_info("push done\n");
        if (trans.done==0) {
            for (int j=0; j<STATE_SIZE; j++)
                trans.state[j] = trans.next_state[j];
        }
        else {
            call_reset(&trans, pinst);
        }
    }
    bsg_pr_test_info("Replay memory init done!!\n");

    // Training loop 
    int num_trans;
    bool re_mem_full = false;
    bool compare_host = true;
    Transition sample_trans;
    float FC1_dW[FC1_W_SIZE], FC1_dB[FC1_B_SIZE];
    float FC2_dW[FC2_W_SIZE], FC2_dB[FC2_B_SIZE];
    float CONV3_dW[CONV3_W_SIZE], CONV3_dB[CONV3_B_SIZE];
    float CONV2_dW[CONV2_W_SIZE], CONV2_dB[CONV2_B_SIZE];
    float CONV1_dW[CONV1_W_SIZE], CONV1_dB[CONV1_B_SIZE];
    float *nn_dw[] = {CONV1_dW, CONV2_dW, CONV3_dW, FC1_dW, FC2_dW};
    float *nn_db[] = {CONV1_dB, CONV2_dB, CONV3_dB, FC1_dB, FC2_dB};
	float host_fc2_w_new[FC2_W_SIZE];
	float host_fc1_w_new[FC1_W_SIZE];
	float host_fc2_b_new[FC2_B_SIZE];
	float host_fc1_b_new[FC1_B_SIZE];
    
    float epsilon = MAX_EPSILON;
    int total_step = 0;
    int step = 0;
    float step_mean = 0.0;
    bool episode_done = false;
	for (int episode = 1; episode < EPISODE_MAX; episode++) {
        episode_done = false;
        step = 0;
        while (!episode_done) {
            total_step++;
            step++;
            bsg_pr_test_info("Step%d-------------------------------------\n", step);
            // Perform one step
            /* bsg_pr_test_info("Perform one step\n"); */
            dqn_act(mc, drlp_dram_eva, &trans, nn, num_layer, epsilon);
            /* bsg_pr_test_info("read conv3 output\n"); */
			/* read_fc_db(mc, FC1_dB, CONV2); */
            /* bsg_pr_test_info("read fc1 output\n"); */
			/* read_fc_db(mc, FC1_dB, CONV3); */
			/* read_drlp_dram(mc, drlp_dram_npa, FC1_WGT_ADDR, 100, FC1_dB, true); */
            call_step(&trans, pinst);

            // Push to replay memory 
            /* bsg_pr_test_info("Push to replay memory\n"); */
	    	position = re_mem_push(mc, re_mem_npa, &trans, position);
            if (position == 0)
                re_mem_full = true;
            if (re_mem_full)
                num_trans = RE_MEM_SIZE;
            else
                num_trans = position+1;

	    	if (trans.done == 0.0) {
	    		for (int j=0; j<STATE_SIZE; j++)
	    			trans.state[j] = trans.next_state[j];
	    	}
	    	else {
	    		call_reset(&trans, pinst);
                episode_done = true;
                step_mean += step;
                if (episode%20==0) {
		            bsg_pr_test_info("Episode: %d, epsilon: %f, mean score: %f\n", episode, epsilon, step_mean/20.0);
                    step_mean = 0.0;
                    float qv[2];
	                nn_fp(mc, drlp_dram_eva, trans.state, nn, num_layer, qv);
		            bsg_pr_test_info("Q[0]: %f\tQ[1]: %f \n", qv[0], qv[1]);
                }
	    	}

            // Training 
	    	if ((total_step%TRAIN_FREQ==0) && (episode_done==false)) {
                // Weight transpose and write
                bsg_pr_test_info("Weight transpose and write\n");
                fc_bp_wrt_wgt(mc, drlp_dram_eva, FC2, FC2_W);
                fc_bp_wrt_wgt(mc, drlp_dram_eva, FC1, FC1_W);
                conv_bp_wrt_wgt(mc, drlp_dram_eva, CONV3, CONV3_W);
                conv_bp_wrt_wgt(mc, drlp_dram_eva, CONV2, CONV2_W);
                conv_bp_wrt_wgt(mc, drlp_dram_eva, CONV1, sample_trans.state);

                // Sample from replay memory
                bsg_pr_test_info("Sample from replay memory\n");
                re_mem_sample(mc, re_mem_npa, &sample_trans, num_trans);

                // Train
                bsg_pr_test_info("DQN train\n");
                dqn_train(mc, drlp_dram_eva, &sample_trans, nn, num_layer, FC2_dB, 0.95);
                read_fc_dw(mc, drlp_dram_eva, FC2_dW, FC2);
                /* read_fc_dw(mc, drlp_dram_npa, FC1_dW, FC1); */
				/* read_fc_db(mc, FC1_dB, FC1); */
                return HB_MC_SUCCESS;
                // if (HOST_COMPARE) {
                //     rc = host_train(sample_trans.state, sample_trans.next_state, sample_trans.reward, sample_trans.done, FC1_W, FC1_B, FC2_W, FC2_B, FC2_WT, FC2_dW, FC1_dW, STATE_SIZE, FC1_Y_SIZE, ACTION_SIZE); 
                //     if (rc==1)
                //         bsg_pr_err("Step%d, BP has error!\n", step);
                //     host_optimizer(host_fc2_w_new, FC2_W, FC2_dW, LR, FC2_W_SIZE);
                //     host_optimizer(host_fc1_w_new, FC1_W, FC1_dW, LR, FC1_W_SIZE);
                // }

                // Optimizer
                for (int i = 0; i < num_layer; i++) {
                    cuda_optimizer(device, bin_path,w_opt_eva[i], dw_opt_eva[i], w_new_opt_eva[i], nn_w[i], nn_dw[i], nn[i].weight_size, LR);
                    cuda_optimizer(device, bin_path,b_opt_eva[i], db_opt_eva[i], b_new_opt_eva[i], nn_b[i], nn_db[i], nn[i].bias_size, LR);
                }

                if (HOST_COMPARE) {
                    rc = host_compare(host_fc2_w_new, FC2_W, FC2_W_SIZE);
                    rc = host_compare(host_fc1_w_new, FC1_W, FC1_W_SIZE);
                    if (rc==1)
                        bsg_pr_err("Step%d, optimizer has error!\n", step);
                }
                // Write new weight to DRAM
                base_addr = FC1_WGT_ADDR;
                fc_fp_wrt_wgt(mc, drlp_dram_eva, FC1, FC1_W, FC1_B, base_addr);
                base_addr = FC2_WGT_ADDR;
                fc_fp_wrt_wgt(mc, drlp_dram_eva, FC2, FC2_W, FC2_B, base_addr);

                if (epsilon*EPSILON_DECAY > MIN_EPSILON)
                    epsilon *= EPSILON_DECAY;
                else
                    epsilon = MIN_EPSILON;

	            return HB_MC_SUCCESS;
            }
        }
    }
    
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
    int rc = test_drlp_dqn_all(argc, argv);
    *exit_code = rc;
    bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
    return;
}
#else
int main(int argc, char ** argv) {
    bsg_pr_test_info(TEST_NAME " Regression Test (F1)\n");
    int rc = test_drlp_dqn_all(argc, argv);
    bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
    return rc;
}
#endif
