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

int cuda_optimizer (hb_mc_device_t device, char *bin_path, float *w, float *dw, int w_num, float lr) {
    /*****************************************************************************************************************
    * CUDA optimizer 
    ******************************************************************************************************************/
    hb_mc_manycore_t *mc = device.mc;
    int rc;
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
    eva_t w_device, dw_device, w_new_device; 
    rc = hb_mc_device_malloc(&device, w_num * sizeof(uint32_t), &w_device); 
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to allocate memory on device.\n");
            return rc;
    }
    rc = hb_mc_device_malloc(&device, w_num * sizeof(uint32_t), &dw_device); 
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to allocate memory on device.\n");
            return rc;
    }
    rc = hb_mc_device_malloc(&device, w_num * sizeof(uint32_t), &w_new_device); 
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to allocate memory on device.\n");
            return rc;
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
    bsg_pr_test_info("========Enqueue and excute cuda kernel========\n");
    rc = hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_optimizer", 6, cuda_argv);
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to initialize grid.\n");
            return rc;
    }

    /* Launch and execute all tile groups on device and wait for all to finish.  */
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


int test_drlp_dqn_all (int argc, char **argv) {

    bsg_pr_test_info("Running DRLP-CUDA DQN test!\n");

    /*****************************************************************************************************************
    * Test game and python settings
    ******************************************************************************************************************/
    char *game_name="Breakout-v0";
    PyObject *pinst;
    pinst = py_init(game_name); // Initialize python class instance and method

    /*****************************************************************************************************************
    * Initialize device 
    ******************************************************************************************************************/
    int rc;
    char *bin_path;
    bin_path = "/mnt/users/ssd1/homes/huwan/bsg/bsg_bladerunner/bsg_manycore/software/spmd/bsg_cuda_lite_runtime/drlp_cuda/main.riscv";
    hb_mc_device_t device;
    rc = hb_mc_device_init(&device, TEST_NAME, 0);
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to initialize device.\n");
            return rc;
    }
    hb_mc_manycore_t *mc = device.mc;

    /*****************************************************************************************************************
    * NN configuration 
    ******************************************************************************************************************/
    int num_layer = 5;
    NN_layer CONV1 = {.input_size=STATE_SIZE, 
                      .output_size=CONV1_Y_SIZE,
                      .weight_size=CONV1_W_SIZE,
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
                    .input_src=ONCHIP,
                    .output_dst=DMA,
                    .relu=0,
                    .layer=4,
                    .FC_CONV=1,
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
    * Weight random initialization and write to dram
    ******************************************************************************************************************/
    // On the host
    float CONV1_W[CONV1_W_SIZE];
    float CONV1_B[32];
    float CONV2_W[CONV2_W_SIZE];
    float CONV2_B[64];
    float CONV3_W[CONV3_W_SIZE];
    float CONV3_B[64];
    float FC1_W[FC1_W_SIZE];
    float FC1_B[FC1_Y_SIZE];
    float FC2_W[FC2_W_SIZE];
    float FC2_B[FC2_Y_SIZE];
    bsg_pr_test_info("Generate weights randomly\n");
    srand(0.1); 
    param_random(CONV1_W, CONV1_W_SIZE);
    param_random(CONV1_B, 32);
    param_random(CONV2_W, CONV2_W_SIZE);
    param_random(CONV2_B, 64);
    param_random(CONV3_W, CONV3_W_SIZE);
    param_random(CONV3_B, 64);
    param_random(FC1_W, FC1_W_SIZE);
    param_random(FC1_B, FC1_Y_SIZE);
    param_random(FC2_W, FC2_W_SIZE);
    param_random(FC2_B, FC2_Y_SIZE);
    // To the device DRAM
    bsg_pr_test_info("Write weights to DRAM\n");
    uint32_t base_addr = CONV1_WGT_ADDR;
    // conv_fp_wrt_wgt(mc, CONV1, CONV1_W, CONV1_B, base_addr);
    // bsg_pr_test_info("CONV222\n");
    // base_addr = CONV2_WGT_ADDR;
    // conv_fp_wrt_wgt(mc, CONV2, CONV2_W, CONV2_B, base_addr);
    // bsg_pr_test_info("CONV333\n");
    // base_addr = CONV3_WGT_ADDR;
    // conv_fp_wrt_wgt(mc, CONV3, CONV3_W, CONV3_B, base_addr);
    // bsg_pr_test_info("FC11\n");
    // base_addr = FC1_WGT_ADDR;
    // fc_fp_wrt_wgt(mc, FC1, FC1_W, FC1_B, base_addr);
    // bsg_pr_test_info("FC22\n");
    // base_addr = FC2_WGT_ADDR;
    // fc_fp_wrt_wgt(mc, FC2, FC2_W, FC2_B, base_addr);

    // bsg_pr_test_info("Write weights to DRAM done!!!\n");

    float FC2_WT[FC2_W_SIZE];
    float FC1_WT[FC1_W_SIZE];
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
        position = re_mem_push(mc, &trans, position);
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

    /* read_re_mem(mc, 0, 40);  */
    
    // for (int i = 0; i < RE_MEM_INIT_SIZE; i++) {
    //     rc = re_mem_sample(mc, &trans, RE_MEM_INIT_SIZE);
    //     for (int j=0; j<STATE_SIZE; j++) {
    //         printf("State[%d]=%1.4f\t", j, trans.state[j]);
    //     }
    //     printf("\n");
    //     for (int j=0; j<STATE_SIZE; j++) {
    //         printf("NextState[%d]=%1.4f\t", j, trans.next_state[j]);
    //     }
    //     printf("\n");
    //     printf("Action=%d\n", trans.action);
    //     printf("Reward=%1.1f\n", trans.reward);
    //     printf("Done=%d\n", trans.done);
    // }

    // Training loop 
    float epsilon=0.0;
    int num_trans;
    bool re_mem_full = false;
    bool compare_host = true;
    Transition sample_trans;
    float FC1_dW[FC1_W_SIZE], FC1_dB[FC1_Y_SIZE];
    float FC2_dW[FC2_W_SIZE], FC2_dB[FC2_Y_SIZE];
    float CONV3_dW[CONV3_W_SIZE], CONV3_dB[64];
    float CONV2_dW[CONV2_W_SIZE], CONV2_dB[64];
    float CONV1_dW[CONV1_W_SIZE], CONV1_dB[32];
    float host_fc2_w_new[FC2_W_SIZE];
    float host_fc1_w_new[FC1_W_SIZE];
    for (int step = 0; step < STEP_MAX; step++) {
        bsg_pr_test_info("Step%d\n", step);

        // Perform one step
        // bsg_pr_test_info("Perform one step\n");
        // dqn_act(mc, &trans, nn, num_layer, epsilon);
        // call_step(&trans, pinst);

        // // Push to replay memory 
        // bsg_pr_test_info("Push to replay memory\n");
        // position = re_mem_push(mc, &trans, position);
        if (position == 0)
            re_mem_full = true;
        if (re_mem_full)
            num_trans = RE_MEM_SIZE;
        else
            num_trans = position+1;

        // if (trans.done==0) {
        //     for (int j=0; j<STATE_SIZE; j++)
        //         trans.state[j] = trans.next_state[j];
        // }
        // else {
        //     call_reset(&trans, pinst);
        // }

        // Training 
        if (step%TRAIN_FREQ==0) {
            // Sample from replay memory
            bsg_pr_test_info("Sample from replay memory\n");
            re_mem_sample(mc, &sample_trans, num_trans);

            // Weight transpose and write
            bsg_pr_test_info("Weight transpose and write\n");
            bsg_pr_test_info("fc2\n");
            fc_bp_wrt_wgt(mc, FC2, FC2_W, FC2_WT);
            /* wgt_transpose_and_write(mc, FC2, FC2_W, FC2_WT); */
            bsg_pr_test_info("fc1\n");
            fc_bp_wrt_wgt(mc, FC1, FC1_W, FC1_WT);
            /* wgt_transpose_and_write(mc, FC1, FC1_W, FC1_WT); */
            bsg_pr_test_info("conv3\n");
            conv_bp_wrt_wgt(mc, CONV3, CONV3_W);
            bsg_pr_test_info("conv2\n");
            conv_bp_wrt_wgt(mc, CONV2, CONV2_W);
            bsg_pr_test_info("conv1\n");
            conv_bp_wrt_wgt(mc, CONV1, sample_trans.state);
            bsg_pr_test_info("conv1 done\n");

            // Train
            dqn_train(mc, &sample_trans, nn, num_layer, 0.95);
            read_dw(mc, FC2_dW, FC2BP_DW_ADDR+1, FC1_Y_SIZE, ACTION_SIZE);
            read_dw(mc, FC1_dW, FC1BP_DW_ADDR+1, STATE_SIZE, FC1_Y_SIZE);
            if (HOST_COMPARE) {
                rc = host_train(sample_trans.state, sample_trans.next_state, sample_trans.reward, sample_trans.done, FC1_W, FC1_B, FC2_W, FC2_B, FC2_WT, FC2_dW, FC1_dW, STATE_SIZE, FC1_Y_SIZE, ACTION_SIZE); 
                if (rc==1)
                    bsg_pr_err("Step%d, BP has error!\n", step);
                host_optimizer(host_fc2_w_new, FC2_W, FC2_dW, LR, FC2_W_SIZE);
                host_optimizer(host_fc1_w_new, FC1_W, FC1_dW, LR, FC1_W_SIZE);
            }

            // Optimizer
            cuda_optimizer(device, bin_path, FC2_W, FC2_dW, FC2_W_SIZE, LR);
            cuda_optimizer(device, bin_path, FC1_W, FC1_dW, FC1_W_SIZE, LR);

            if (HOST_COMPARE) {
                rc = host_compare(host_fc2_w_new, FC2_W, FC2_W_SIZE);
                rc = host_compare(host_fc1_w_new, FC1_W, FC1_W_SIZE);
                if (rc==1)
                    bsg_pr_err("Step%d, optimizer has error!\n", step);
            }
            // Write new weight to DRAM
            base_addr = FC1_WGT_ADDR;
            fc_fp_wrt_wgt(mc, FC1, FC1_W, FC1_B, base_addr);
            base_addr = FC2_WGT_ADDR;
            fc_fp_wrt_wgt(mc, FC2, FC2_W, FC2_B, base_addr);

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
