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

int cuda_optimizer (hb_mc_device_t device, NN_layer nn, hb_mc_eva_t base_eva, float lr) {
    int rc;
    eva_t w_eva = base_eva + (nn.wgt_base_addr<<2);
    eva_t wT_eva = base_eva + (nn.wT_base_addr<<2);
    eva_t dw_eva = base_eva + ((nn.dw_base_addr+1)<<2);
    eva_t db_eva = base_eva + (nn.db_base_addr<<2);
    int w_num = nn.weight_size;
    
    /* Define block_size_x/y: amount of work for each tile group */
    /* Define tg_dim_x/y: number of tiles in each tile group */
    /* Calculate grid_dim_x/y: number of tile groups needed based on block_size_x/y */
    uint32_t block_size_x = w_num;
    hb_mc_dimension_t tg_dim = { .x = 2, .y = 2}; 
    hb_mc_dimension_t grid_dim = { .x = 1, .y = 1}; 

    /* Prepare list of input arguments for kernel. */
    int cuda_argv[17] = {2, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
        w_eva, wT_eva, dw_eva, db_eva, nn.layer, w_num, block_size_x};

    /* Enquque grid of tile groups, pass in grid and tile group dimensions, kernel name, number and list of input arguments */
    rc = hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_optimizer", 17, cuda_argv);
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

}

int cuda_re_mem (hb_mc_device_t device, int flag,
        hb_mc_eva_t re_mem_state_eva, hb_mc_eva_t re_mem_next_state_eva, hb_mc_eva_t re_mem_reward_eva, hb_mc_eva_t re_mem_action_eva, hb_mc_eva_t re_mem_done_eva, 
        Transition *trans, uint32_t position) {
    int rc;

    hb_mc_eva_t trans_state_eva, trans_next_state_eva, trans_others_eva, trans_action_eva, trans_done_eva;
    rc = hb_mc_device_malloc(&device, sizeof(float)*STATE_SIZE, &trans_state_eva); 
    rc = hb_mc_device_malloc(&device, sizeof(float)*STATE_SIZE, &trans_next_state_eva); 
    rc = hb_mc_device_malloc(&device, sizeof(float)*128, &trans_others_eva); 

    void *dst, *src;
    if (flag == 0) {
        dst = (void *) ((intptr_t) trans_state_eva);
        src = (void *) &(trans->state[0]);
        rc = hb_mc_device_memcpy (&device, dst, src, STATE_SIZE * sizeof(float), HB_MC_MEMCPY_TO_DEVICE);     

        dst = (void *) ((intptr_t) trans_next_state_eva);
        src = (void *) &(trans->next_state[0]);
        rc = hb_mc_device_memcpy (&device, dst, src, STATE_SIZE * sizeof(float), HB_MC_MEMCPY_TO_DEVICE);     

        float others[4] = {trans->reward, trans->action, trans->done};
        dst = (void *) ((intptr_t) trans_others_eva);
        src = (void *) &(others[0]);
        rc = hb_mc_device_memcpy (&device, dst, src, sizeof(float)*4, HB_MC_MEMCPY_TO_DEVICE);
    }

    
    hb_mc_dimension_t tg_dim = { .x = 2, .y = 2}; 
    hb_mc_dimension_t grid_dim = { .x = 1, .y = 1}; 

    int cuda_argv[17] = {flag, 
        re_mem_state_eva, re_mem_next_state_eva, 
        re_mem_reward_eva, re_mem_action_eva, re_mem_done_eva,
        position, trans_state_eva, trans_next_state_eva, trans_others_eva, 
        1024, 1024, 1024, 1024, 0, 64, 64};


    /* bsg_pr_test_info("cuda_re_mem enqueue: flag=%d, position=%d \n",flag, position); */
    rc = hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_optimizer", 17, cuda_argv);
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to initialize grid.\n");
    }

    rc = hb_mc_device_tile_groups_execute(&device);
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to execute tile groups.\n");
    }

    if (flag == 1) {
        src = (void *) ((intptr_t) trans_state_eva);
        dst = (void *) &(trans->state[0]);
        rc = hb_mc_device_memcpy (&device, (void *) dst, src, sizeof(float)*STATE_SIZE, HB_MC_MEMCPY_TO_HOST);

        src = (void *) ((intptr_t) trans_next_state_eva);
        dst = (void *) &(trans->next_state[0]);
        rc = hb_mc_device_memcpy (&device, (void *) dst, src, sizeof(float)*STATE_SIZE, HB_MC_MEMCPY_TO_HOST);

        float aaa[4];
        src = (void *) ((intptr_t) trans_others_eva);
        dst = (void *) &aaa[0];
        rc = hb_mc_device_memcpy (&device, (void *) dst, src, sizeof(float)*4, HB_MC_MEMCPY_TO_HOST);
        if (rc != HB_MC_SUCCESS) { 
             bsg_pr_err("failed to reward.\n");
        }
        trans->reward = aaa[0];
        trans->action = aaa[1];
        trans->done = aaa[2];

    }
    
    if ((position+1)==RE_MEM_SIZE)
        return 0;
    else
        return position+1;
}


int test_drlp_dqn_all (int argc, char **argv) {

    bsg_pr_test_info("Running DRLP-CUDA DQN Breakout test!\n");

    double time_spent = 0.0;
    clock_t begin = clock();

    srand(0.1); 

    /*****************************************************************************************************************
    * Test game and python settings
    ******************************************************************************************************************/
    char *game_name="Breakout-v0";
    PyObject *py_game;
    py_game = py_init(game_name); // Initialize python class instance and method

    PyObject *py_dqn;
    py_dqn = py_init("torch_dqn");

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
                      .dw_base_addr=CONV1BP_DW_ADDR,
                      .db_base_addr=CONV1BP_DB_ADDR
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
                      .db_base_addr=CONV2BP_DB_ADDR,
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
                      .db_base_addr=CONV3BP_DB_ADDR,
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
                    .db_base_addr=FC1BP_DB_ADDR,
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
                    .db_base_addr=FC2BP_DB_ADDR,
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

    /*****************************************************************************************************************
    * Memory allocation 
    ******************************************************************************************************************/
    /* Allocate memory on the device for DRLP operation*/
    size_t drlp_dram_size = sizeof(uint32_t)*DRLP_DRAM_SIZE;
    hb_mc_eva_t drlp_dram_eva;
    hb_mc_device_malloc(&device, drlp_dram_size, &drlp_dram_eva); 
    write_dram_configure(mc, drlp_dram_eva>>2);
    bsg_pr_test_info("DRLP dram eva: %x\n", drlp_dram_eva);

    /* Allocate memory on the device for replay memory*/
    hb_mc_eva_t re_mem_state_eva, re_mem_next_state_eva, re_mem_reward_eva, re_mem_action_eva, re_mem_done_eva;
    hb_mc_device_malloc(&device, sizeof(float)*STATE_SIZE*RE_MEM_SIZE, &re_mem_state_eva); 
    bsg_pr_test_info("state eva: %x\n", re_mem_state_eva);
    hb_mc_device_malloc(&device, sizeof(float)*STATE_SIZE*RE_MEM_SIZE, &re_mem_next_state_eva); 
    bsg_pr_test_info("next state eva: %x\n", re_mem_next_state_eva);

    rc = hb_mc_device_malloc(&device, sizeof(float)*RE_MEM_SIZE, &re_mem_reward_eva); 
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to initialize reward.\n");
            return rc;
    }
    bsg_pr_test_info("reward eva: %x\n", re_mem_reward_eva);

    hb_mc_device_malloc(&device, sizeof(uint32_t)*RE_MEM_SIZE, &re_mem_action_eva); 
    bsg_pr_test_info("action eva: %x\n", re_mem_action_eva);
    hb_mc_device_malloc(&device, sizeof(uint32_t)*RE_MEM_SIZE, &re_mem_done_eva); 
    bsg_pr_test_info("done eva: %x\n", re_mem_done_eva);


    /*****************************************************************************************************************
    * Weight random initialization and write to dram
    ******************************************************************************************************************/
    // On the host
    bsg_pr_test_info("Initialize weights randomly and write to DRAM\n");
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

    get_parameters(py_dqn, nn_w, nn_b, false);

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

    // Weight transpose and write
    bsg_pr_test_info("Weight transpose and write\n");
    fc_bp_wrt_wgt(mc, drlp_dram_eva, FC2, FC2_W);
    fc_bp_wrt_wgt(mc, drlp_dram_eva, FC1, FC1_W);
    conv_bp_wrt_wgt(mc, drlp_dram_eva, CONV3, CONV3_W);
    conv_bp_wrt_wgt(mc, drlp_dram_eva, CONV2, CONV2_W);

    /*****************************************************************************************************************
    * DQN  
    ******************************************************************************************************************/
    // Replay memory init
    bsg_pr_test_info("Replay memory initialization\n");
    uint32_t position = 0;
    Transition trans;
    call_reset(&trans, py_game);
    for (int i = 0; i < RE_MEM_INIT_SIZE; i++) {
        trans.action = rand() % ACTION_SIZE;
        call_step(&trans, py_game);
        /* position = re_mem_push(mc, re_mem_npa, &trans, position); */
        position = cuda_re_mem(device, 0, re_mem_state_eva, re_mem_next_state_eva, re_mem_reward_eva, re_mem_action_eva, re_mem_done_eva, &trans, position);
        if (trans.done==0) {
            for (int j=0; j<STATE_SIZE; j++)
                trans.state[j] = trans.next_state[j];
        }
        else {
            call_reset(&trans, py_game);
        }
    }

    // Training loop 
    int num_trans;
    bool re_mem_full = false;
    Transition sample_trans;
    
    float epsilon = MAX_EPSILON;
    int total_step = 0;
    int step = 0;
    float step_mean = 0.0;
    bool episode_done = false;
    for (int episode = 1; episode < EPISODE_MAX; episode++) {
        bsg_pr_test_info("========Episode %d========\n", episode);
        episode_done = false;
        step = 0;
        while (!episode_done) {
            clock_t step_start = clock();
            double step_time_spent = 0.0;

            total_step++;
            step++;
            bsg_pr_test_info("Frame %d\n", step);
            // Perform one step
            bsg_pr_test_info("Perform one action and store the trainsition into replay memory\n");
            float drlp_fp_r[ACTION_SIZE];
            dqn_act(mc, drlp_dram_eva, &trans, nn, num_layer, epsilon, drlp_fp_r);
            if (HOST_COMPARE) {
                float torch_fp_r[ACTION_SIZE];
                torch_forward(&trans, py_dqn, torch_fp_r); 
                host_compare(torch_fp_r, drlp_fp_r, ACTION_SIZE, "fp");
            }

            call_step(&trans, py_game);

            // Push to replay memory 
            /* position = re_mem_push(mc, re_mem_npa, &trans, position); */
            position = cuda_re_mem(device, 0, re_mem_state_eva, re_mem_next_state_eva, re_mem_reward_eva, re_mem_action_eva, re_mem_done_eva, &trans, position);
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
                call_reset(&trans, py_game);
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
                // Sample from replay memory
                bsg_pr_test_info("Sample one transition from replay memory\n");
                /* re_mem_sample(mc, re_mem_npa, &sample_trans, num_trans); */
                cuda_re_mem(device, 1, re_mem_state_eva, re_mem_next_state_eva, re_mem_reward_eva, re_mem_action_eva, re_mem_done_eva, &sample_trans, 1);

                // Write state for first layer BP
                conv_bp_wrt_wgt(mc, drlp_dram_eva, CONV1, sample_trans.state);

                // Train
                bsg_pr_test_info("DQN train\n");
                static float FC1_dW[FC1_W_SIZE], FC1_dB[FC1_B_SIZE];
                static float FC2_dW[FC2_W_SIZE], FC2_dB[FC2_B_SIZE];
                static float CONV3_dW[CONV3_W_SIZE], CONV3_dB[CONV3_B_SIZE];
                static float CONV2_dW[CONV2_W_SIZE], CONV2_dB[CONV2_B_SIZE];
                static float CONV1_dW[CONV1_W_SIZE], CONV1_dB[CONV1_B_SIZE];
                float *nn_dw[] = {CONV1_dW, CONV2_dW, CONV3_dW, FC1_dW, FC2_dW};
                float *nn_db[] = {CONV1_dB, CONV2_dB, CONV3_dB, FC1_dB, FC2_dB};
                dqn_train(mc, drlp_dram_eva, &sample_trans, nn, num_layer, FC2_dB, 0.95);

                // Optimizer
                cuda_optimizer(device, FC2, drlp_dram_eva, LR);
                cuda_optimizer(device, FC1, drlp_dram_eva, LR);
                cuda_optimizer(device, CONV3, drlp_dram_eva, LR);
                cuda_optimizer(device, CONV2, drlp_dram_eva, LR);
                cuda_optimizer(device, CONV1, drlp_dram_eva, LR);

                clock_t step_end = clock();
                step_time_spent += (double)(step_end - step_start) / CLOCKS_PER_SEC;
                printf("Time elapsed is %f seconds\n", step_time_spent);

                if (HOST_COMPARE) {
                    static float HOST_FC1_dW[FC1_W_SIZE], HOST_FC1_dB[FC1_B_SIZE];
                    static float HOST_FC2_dW[FC2_W_SIZE],  HOST_FC2_dB[FC2_B_SIZE];
                    static float HOST_CONV3_dW[CONV3_W_SIZE], HOST_CONV3_dB[CONV3_B_SIZE];
                    static float HOST_CONV2_dW[CONV2_W_SIZE], HOST_CONV2_dB[CONV2_B_SIZE];
                    static float HOST_CONV1_dW[CONV1_W_SIZE], HOST_CONV1_dB[CONV1_B_SIZE];
                    float *host_nn_dw[] = {HOST_CONV1_dW, HOST_CONV2_dW, HOST_CONV3_dW, HOST_FC1_dW, HOST_FC2_dW};
                    float *host_nn_db[] = {HOST_CONV1_dB, HOST_CONV2_dB, HOST_CONV3_dB, HOST_FC1_dB, HOST_FC2_dB};
                    static float HOST_FC1_W[FC1_W_SIZE], HOST_FC1_B[FC1_B_SIZE];
                    static float HOST_FC2_W[FC2_W_SIZE], HOST_FC2_B[FC2_B_SIZE];
                    static float HOST_CONV3_W[CONV3_W_SIZE], HOST_CONV3_B[CONV3_B_SIZE];
                    static float HOST_CONV2_W[CONV2_W_SIZE], HOST_CONV2_B[CONV2_B_SIZE];
                    static float HOST_CONV1_W[CONV1_W_SIZE], HOST_CONV1_B[CONV1_B_SIZE];
                    float *host_nn_w[] = {HOST_CONV1_W, HOST_CONV2_W, HOST_CONV3_W, HOST_FC1_W, HOST_FC2_W};
                    float *host_nn_b[] = {HOST_CONV1_B, HOST_CONV2_B, HOST_CONV3_B, HOST_FC1_B, HOST_FC2_B};

                    // compare dw
                    bsg_pr_test_info("Compare gradients with PyTorch\n");
                    torch_train(&sample_trans, py_dqn, host_nn_dw, host_nn_db);
                    read_fc_dw(mc, drlp_dram_eva, FC2_dW, FC2);
                    read_fc_dw(mc, drlp_dram_eva, FC1_dW, FC1);
                    read_conv_dw(mc, drlp_dram_eva, CONV3_dW, CONV3);
                    read_conv_dw(mc, drlp_dram_eva, CONV2_dW, CONV2);
                    read_conv_dw(mc, drlp_dram_eva, CONV1_dW, CONV1);
                    host_compare(HOST_FC2_dW, FC2_dW, FC2_W_SIZE, "fc2_dw");
                    host_compare(HOST_FC1_dW, FC1_dW, FC1_W_SIZE, "fc1_dw");
                    host_compare(HOST_CONV3_dW, CONV3_dW, CONV3_W_SIZE, "conv3_dw");
                    host_compare(HOST_CONV2_dW, CONV2_dW, CONV2_W_SIZE, "conv2_dw");
                    host_compare(HOST_CONV1_dW, CONV1_dW, CONV1_W_SIZE, "conv1_dw");

                    // compare new w for fp
                    bsg_pr_test_info("Compare new weight with PyTorch\n");
                    get_parameters(py_dqn, host_nn_w, host_nn_b, false);
                    read_fc_w(mc, drlp_dram_eva, FC2, FC2_W, FC2_B);
                    read_fc_w(mc, drlp_dram_eva, FC1, FC1_W, FC1_B);
                    read_conv_w(mc, drlp_dram_eva, CONV3, CONV3_W, CONV3_B);
                    read_conv_w(mc, drlp_dram_eva, CONV2, CONV2_W, CONV2_B);
                    read_conv_w(mc, drlp_dram_eva, CONV1, CONV1_W, CONV1_B);
                    host_compare(HOST_FC2_W, FC2_W, FC2_W_SIZE, "fc2 new W for FP");
                    host_compare(HOST_FC1_W, FC1_W, FC1_W_SIZE, "fc1 new W for FP");
                    host_compare(HOST_CONV3_W, CONV3_W, CONV3_W_SIZE, "conv3 new W for FP");
                    host_compare(HOST_CONV2_W, CONV2_W, CONV2_W_SIZE, "conv2 new W for FP");
                    host_compare(HOST_CONV1_W, CONV1_W, CONV1_W_SIZE, "conv1 new W for FP");
                    host_compare(HOST_FC2_B, FC2_B, FC2.bias_size, "fc2 new B");
                    host_compare(HOST_FC1_B, FC1_B, FC2.bias_size, "fc1 new B");
                    host_compare(HOST_CONV3_B, CONV3_B, CONV3.bias_size, "conv3 new B");
                    host_compare(HOST_CONV2_B, CONV2_B, CONV2.bias_size, "conv2 new B");
                    host_compare(HOST_CONV1_B, CONV1_B, CONV1.bias_size, "conv1 new B");
                    // compare new w for bp
                    read_fc_wT(mc, drlp_dram_eva, FC2, FC2_W);
                    read_fc_wT(mc, drlp_dram_eva, FC1, FC1_W);
                    read_conv_wT(mc, drlp_dram_eva, CONV3, CONV3_W);
                    read_conv_wT(mc, drlp_dram_eva, CONV2, CONV2_W);
                    host_compare(HOST_FC2_W, FC2_W, FC2_W_SIZE, "fc2 new W for BP");
                    host_compare(HOST_FC1_W, FC1_W, FC1_W_SIZE, "fc1 new W for BP");
                    host_compare(HOST_CONV3_W, CONV3_W, CONV3_W_SIZE, "conv2 new W for BP");
                    host_compare(HOST_CONV2_W, CONV2_W, CONV2_W_SIZE, "conv1 new W for BP");
                }

                if (epsilon*EPSILON_DECAY > MIN_EPSILON)
                    epsilon *= EPSILON_DECAY;
                else
                    epsilon = MIN_EPSILON;

            }
        }
    }

    clock_t end = clock();
    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time elapsed is %f seconds\n", time_spent);

    /* Freeze the tiles and memory manager cleanup. */
    Py_DECREF(py_game);    
    Py_DECREF(py_dqn);    
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
