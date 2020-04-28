#include <Python.h>
// #include <numpy/arrayobject.h>
#include </usr/local/lib64/python3.6/site-packages/numpy/core/include/numpy/arrayobject.h>
// #include "test_drlp_fpbp.h"

#ifndef __LIBRARY_TESTS_H
#define __LIBRARY_TESTS_H

#include <bsg_manycore_loader.h>
#include <bsg_manycore_errno.h>    
#include <bsg_manycore_printing.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include "spmd_tests.h"
#include "cuda_tests.h"

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


#include "test_drlp_host_gd.h"
#include "test_drlp_param.h"
#include "test_drlp_struct.h"
#include "test_drlp_alloc_mem.h"
#include "test_drlp_layer_config.h"
#include <math.h>
void param_random(float *param, int size){
    for (int i = 0; i < size; i++)
        if (rand()%2==0)
            param[i] = rand()/(float)(RAND_MAX)/10;
        else
            param[i] = -rand()/(float)(RAND_MAX)/10;
}

/*****************************************************************************************************************
* For Python Calling 
******************************************************************************************************************/

PyObject* py_init(char *game_name) {
    Py_Initialize(); 
    import_array();

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('../../regression/drlp_test/')"); // Add regression directory to PATH
      
    PyObject *pmod, *pclass, *pargs, *pinst;

    char *module_name = malloc(100);
    if (strcmp(game_name, "torch_dqn")==0)
        strcpy(module_name, "torch_dqn");
    else
        strcpy(module_name, "gym_env");

    // Load module
    pmod = PyImport_ImportModule(module_name);
    if (pmod == NULL)
        bsg_pr_err("Can't load module");
    pclass = PyObject_GetAttrString(pmod, module_name);
    Py_DECREF(pmod);

    // Load class
    pargs = Py_BuildValue("(s)", game_name);
    pinst = PyEval_CallObject(pclass, pargs);
    Py_DECREF(pclass);
    Py_DECREF(pargs);

    return pinst;
}

// for torch dqn
void get_parameters(PyObject *pinst, float *nn_w[], float *nn_b[], bool read_grad) { 
    printf("Copy nn parameters from torch\n");
    char *get_w[5];
    char *get_b[5];
    if (!read_grad) {
        get_w[0] = "get_conv1_w";
        get_b[0] = "get_conv1_b";
        get_w[1] = "get_conv2_w";
        get_b[1] = "get_conv2_b";
        get_w[2] = "get_conv3_w";
        get_b[2] = "get_conv3_b";
        get_w[3] = "get_fc1_w";
        get_b[3] = "get_fc1_b";
        get_w[4] = "get_fc2_w";
        get_b[4] = "get_fc2_b";
    }
    else {
        get_w[0]= "get_conv1_dw";
        get_b[0]= "get_conv1_db";
        get_w[1]= "get_conv2_dw";
        get_b[1]= "get_conv2_db";
        get_w[2]= "get_conv3_dw";
        get_b[2]= "get_conv3_db";
        get_w[3]= "get_fc1_dw";
        get_b[3]= "get_fc1_db";
        get_w[4]= "get_fc2_dw";
        get_b[4]= "get_fc2_db";
    }
    for (int i=0; i<5; i++) {
        PyObject *pgetw = PyObject_GetAttrString(pinst, get_w[i]);
        PyObject *pgetb = PyObject_GetAttrString(pinst, get_b[i]);
        PyObject *pargs  = Py_BuildValue("()");
        PyArrayObject *pwgt = PyEval_CallObject(pgetw, pargs);
        PyArrayObject *pbias = PyEval_CallObject(pgetb, pargs);
        Py_DECREF(pargs);
        Py_DECREF(pgetw);
        Py_DECREF(pgetb);
        if (PyArray_Check(pwgt) && PyArray_Check(pbias)) {
            if (i < 3) {
                // read conv layer weights
                int nums = pwgt->dimensions[0];
                int chas = pwgt->dimensions[1];
                int rows = pwgt->dimensions[2];
                int cols = pwgt->dimensions[3];
                int num_step = pwgt->strides[0];
                int cha_step = pwgt->strides[1];
                int row_step = pwgt->strides[2];
                int col_step = pwgt->strides[3];
                // bsg_pr_test_info("chas:%d, cha_step:%d, rows:%d, row_step:%d, cols:%d, col_step:%d\n", chas, cha_step, rows, row_step, cols, col_step);
                for (int n=0; n<nums; n++) {
                    for (int d=0; d<chas; d++) {
                        for (int r=0; r<rows; r++) {
                            for (int c=0; c<cols; c++) {
                                nn_w[i][n*(rows*cols*chas)+d*(rows*cols)+r*cols+c] = *(float*)(pwgt->data + n*num_step + d*cha_step + r*row_step + c*col_step);
                                // printf("state[%d,] is % f\n", r, trans->state[r]);
                            }
                        }
                    }
                }
            }
            else {
                // read fc layer weights
                int rows = pwgt->dimensions[0];
                int cols = pwgt->dimensions[1];
                int row_step = pwgt->strides[0];
                int col_step = pwgt->strides[1];
                // bsg_pr_test_info("rows:%d, row_step:%d, cols:%d, col_step:%d\n", rows, row_step, cols, col_step);
                for (int r=0; r<rows; r++) {
                    for (int c=0; c<cols; c++) {
                        nn_w[i][r+c*rows] = *(float*)(pwgt->data + r*row_step + c*col_step);
                        // printf("state[%d,] is % f\n", r, trans->state[r]);
                    }
                }
            }
            // read bias
            int nums = pbias->dimensions[0];
            int num_step = pbias->strides[0];
            for (int n=0; n<nums; n++) {
                nn_b[i][n] = *(float*)(pbias->data + n*num_step);
            }
        }
        else {
            printf("Returned state is not Numpy array!\n");
        }
        Py_DECREF(pwgt);
        Py_DECREF(pbias);
    }
}

void torch_forward(Transition *trans, PyObject *pinst, float *result) { 
    PyObject *pforward = PyObject_GetAttrString(pinst, "c_call_forward");
    npy_intp dims[1] = {84*84*4};
    PyObject *parray = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, trans->state);
    PyObject *argarray = PyTuple_New(1);
    PyTuple_SetItem(argarray, 0, parray);
    PyArrayObject *poutput = PyObject_CallObject(pforward, argarray);
    if (PyArray_Check(poutput)) {
        int nums = poutput->dimensions[0];
        int num_step = poutput->strides[0];
        for (int n=0; n<nums; n++) {
            result[n] = *(float*)(poutput->data + n*num_step);
        }
    }
    else {
        bsg_pr_err("torch forward returned is not Numpy array!\n");
    }
    Py_DECREF(argarray);
    Py_DECREF(pforward);
}

void torch_train(Transition *trans, PyObject *pinst, float *nn_dw[], float *nn_db[]) { 
    // c_call_train has 5 args: state, next state, reward, action, done
    PyObject *ptrain = PyObject_GetAttrString(pinst, "c_call_train");
    npy_intp dims[1] = {84*84*4};
    PyObject *arg_state = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, trans->state);
    PyObject *args_tuple = PyTuple_New(6);
    PyTuple_SetItem(args_tuple, 0, arg_state);
    PyObject *arg_next_state = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, trans->next_state);
    PyTuple_SetItem(args_tuple, 1, arg_next_state);
    PyTuple_SetItem(args_tuple, 2, Py_BuildValue("f",trans->reward));
    PyTuple_SetItem(args_tuple, 3, Py_BuildValue("i",trans->action));
    PyTuple_SetItem(args_tuple, 4, Py_BuildValue("i",trans->done));
    PyTuple_SetItem(args_tuple, 5, Py_BuildValue("f", 0.95));
    
    PyArrayObject *poutput = PyObject_CallObject(ptrain, args_tuple);
    if (PyArray_Check(poutput)) {
        int nums = poutput->dimensions[0];
        int num_step = poutput->strides[0];
        float result[4];
        for (int n=0; n<nums; n++) {
            result[n] = *(float*)(poutput->data + n*num_step);
            // bsg_pr_test_info("%f\n",result[n]);
        }
    }
    else {
        bsg_pr_err("torch forward returned is not Numpy array!\n");
    }
    Py_DECREF(args_tuple);
    Py_DECREF(ptrain);

    get_parameters(pinst, nn_dw, nn_db, true);
}


// for game environment
void call_reset(Transition *trans, PyObject *pinst) { 
    printf("call reset\n");
    PyObject *preset = PyObject_GetAttrString(pinst, "reset");
    PyObject *pargs  = Py_BuildValue("()");
    PyArrayObject *pstate = PyEval_CallObject(preset, pargs);
    Py_DECREF(pargs);
    Py_DECREF(preset);
    if (PyArray_Check(pstate)) {
        int chas = pstate->dimensions[0];
        int rows = pstate->dimensions[1];
        int cols = pstate->dimensions[2];
        int cha_step = pstate->strides[0];
        int row_step = pstate->strides[1];
        int col_step = pstate->strides[2];
        for (int d=0; d<chas; d++){
            for (int r=0; r<rows; r++){
                for (int c=0; c<cols; c++){
                    trans->state[d*(rows*cols)+r*cols+c] = *(float*)(pstate->data + d*cha_step + r*row_step + c*col_step);
                }
            }
        }
    }
    else {
        printf("Returned state is not Numpy array!\n");
    }
    Py_DECREF(pstate);
    trans->done = 0;
}

void call_step(Transition *trans, PyObject *pinst) { 
    // Copy ths first 3 frames
    for (int d=0; d<3; d++){
        for (int r=0; r<84; r++){
            for (int c=0; c<84; c++){
                trans->next_state[d*84*84+r*84+c] = trans->state[(d+1)*84*84+r*84+c] ;
            }
        }
    }
    PyObject *pstep  = PyObject_GetAttrString(pinst, "step");
    PyObject *pargs  = Py_BuildValue("(i)", trans->action);
    PyArrayObject *pcatall = PyEval_CallObject(pstep, pargs);
    Py_DECREF(pargs);
    Py_DECREF(pstep);
    if (PyArray_Check(pcatall)) {
        int rows = pcatall->dimensions[0];
        int row_step = pcatall->strides[0];
        int cols = pcatall->dimensions[1];
        int col_step = pcatall->strides[1];
        // bsg_pr_test_info("rows:%d, row_step:%d, cols:%d, col_step:%d\n", rows, row_step, cols, col_step);
        // Read next state
        for (int r=0; r<rows-1; r++){
            for (int c=0; c<cols; c++){
                trans->next_state[3*84*84+r*cols+c] = *(float*)(pcatall->data + r*row_step + c*col_step);
                // printf("state[%d,] is % f\n", r, trans->next_state[r]);
            }
        }
        // Read reward and done
        trans->reward = *(float*)(pcatall->data + (rows-1)*row_step);
        // printf("Inside Reward is % f\n", trans->reward);
        trans->done = *(float*)(pcatall->data + (rows-1)*row_step + 1*col_step);
        // printf("Inside Done is %d \n", trans->done);
    }
    else {
        printf("Returned state is not Numpy array!\n");
    }
    Py_DECREF(pcatall);
}

/*****************************************************************************************************************
* For Replay Memory 
******************************************************************************************************************/

uint32_t re_mem_push(hb_mc_manycore_t *mc, hb_mc_npa_t base_npa,  Transition *trans, uint32_t position) {
    hb_mc_idx_t re_mem_x = base_npa.x;
    hb_mc_idx_t re_mem_y = base_npa.y;
    hb_mc_epa_t re_mem_epa = base_npa.epa;

    int rc;
    float number;
    uint32_t num_int;
    uint32_t trans_size = TRANSITION_SIZE;
    uint32_t addr = re_mem_epa + (position*trans_size);
    // State
    for (int i=0; i<STATE_SIZE; i++) {
        number = trans->state[i];
        // bsg_pr_test_info("state[%d]=%f\n",i,number);
        num_int = hex(number);
        hb_mc_npa_t npa = {.x = re_mem_x, .y = re_mem_y, .epa = addr*4};
        rc = hb_mc_manycore_write_mem(mc, &npa, &num_int, sizeof(num_int));
        if (rc != HB_MC_SUCCESS) {
            bsg_pr_err("%s: failed to write state 0x%08" PRIx32 " "
                   "to DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
                   __func__, num_int, re_mem_x, re_mem_y, addr);
            hb_mc_manycore_exit(mc);
        }
        addr++;
    }

    // Next state
    for (int i=0; i<STATE_SIZE; i++) {
        number = trans->next_state[i];
        num_int = hex(number);
        hb_mc_npa_t npa = {.x = re_mem_x, .y = re_mem_y, .epa = addr*4};
        rc = hb_mc_manycore_write_mem(mc, &npa, &num_int, sizeof(num_int));
        if (rc != HB_MC_SUCCESS) {
            bsg_pr_err("%s: failed to write next_state 0x%08" PRIx32 " "
                   "to DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
                   __func__, num_int, re_mem_x, re_mem_y, addr);
            hb_mc_manycore_exit(mc);
        }
        addr++;
    }

    // Action
    num_int = trans->action;
    hb_mc_npa_t npa0 = {.x = re_mem_x, .y = re_mem_y, .epa = addr*4};
    rc = hb_mc_manycore_write_mem(mc, &npa0, &num_int, sizeof(num_int));
    if (rc != HB_MC_SUCCESS) {
        bsg_pr_err("%s: failed to write action 0x%08" PRIx32 " "
               "to DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
               __func__, num_int, re_mem_x, re_mem_y, addr);
        hb_mc_manycore_exit(mc);
    }
    addr++;

    // Reward
    num_int = hex(trans->reward);
    hb_mc_npa_t npa1 = {.x = re_mem_x, .y = re_mem_y, .epa = addr*4};
    rc = hb_mc_manycore_write_mem(mc, &npa1, &num_int, sizeof(num_int));
    if (rc != HB_MC_SUCCESS) {
        bsg_pr_err("%s: failed to write reward 0x%08" PRIx32 " "
               "to DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
               __func__, num_int, re_mem_x, re_mem_y, addr);
        hb_mc_manycore_exit(mc);
    }
    addr++;

    // Done
    num_int = trans->done;
    hb_mc_npa_t npa2 = {.x = re_mem_x, .y = re_mem_y, .epa = addr*4};
    rc = hb_mc_manycore_write_mem(mc, &npa2, &num_int, sizeof(num_int));
    if (rc != HB_MC_SUCCESS) {
        bsg_pr_err("%s: failed to write done 0x%08" PRIx32 " "
               "to DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
               __func__, num_int, re_mem_x, re_mem_y, addr);
        hb_mc_manycore_exit(mc);
    }

    if ((position+1)==RE_MEM_SIZE)
        return 0;
    else
        return position+1;
}

int re_mem_sample(hb_mc_manycore_t *mc, hb_mc_npa_t base_npa, Transition *trans, uint32_t size) {
    hb_mc_idx_t re_mem_x = base_npa.x;
    hb_mc_idx_t re_mem_y = base_npa.y;
    hb_mc_epa_t re_mem_epa = base_npa.epa;
    uint32_t position = (rand()%size);
    uint32_t trans_size = TRANSITION_SIZE;
    uint32_t addr = re_mem_epa + (position*trans_size);
    uint32_t num_int;
    int err;
    // printf("Sample %dth transition from replay memory addr%d.\n", position, addr);

    // State
    for (int i=0; i<STATE_SIZE; i++) {
        hb_mc_npa_t npa = { .x = re_mem_x, .y = re_mem_y, .epa = addr*4 };
        err = hb_mc_manycore_read_mem(mc, &npa, &num_int, sizeof(num_int));
        trans->state[i] = flt(num_int);
        if (err != HB_MC_SUCCESS) {
            bsg_pr_err("%s: failed to read from replay memory: %s\n", __func__, hb_mc_strerror(err));
            hb_mc_manycore_exit(mc);
            return err;
        }
        // printf("Read state(%d)%f from epa(%d).\n", i, flt(num_int), addr*4);
        addr++;
    }

    // Next state
    for (int i=0; i<STATE_SIZE; i++) {
        hb_mc_npa_t npa = { .x = re_mem_x, .y = re_mem_y, .epa = addr*4 };
        err = hb_mc_manycore_read_mem(mc, &npa, &num_int, sizeof(num_int));
        trans->next_state[i] = flt(num_int);
        if (err != HB_MC_SUCCESS) {
            bsg_pr_err("%s: failed to read from replay memory: %s\n", __func__, hb_mc_strerror(err));
            hb_mc_manycore_exit(mc);
            return err;
        }
        addr++;
    }

    // Action
    hb_mc_npa_t npa0 = {.x = re_mem_x, .y = re_mem_y, .epa = addr*4};
    err = hb_mc_manycore_read_mem(mc, &npa0, &num_int, sizeof(num_int));
    if (err != HB_MC_SUCCESS) {
        bsg_pr_err("%s: failed to read from replay memory: %s\n", __func__, hb_mc_strerror(err));
        hb_mc_manycore_exit(mc);
        return err;
    }
    trans->action = num_int;
    addr++;

    // Reward
    hb_mc_npa_t npa1 = {.x = re_mem_x, .y = re_mem_y, .epa = addr*4};
    err = hb_mc_manycore_read_mem(mc, &npa1, &num_int, sizeof(num_int));
    if (err != HB_MC_SUCCESS) {
        bsg_pr_err("%s: failed to read from manycore DMEM: %s\n", __func__, hb_mc_strerror(err));
        hb_mc_manycore_exit(mc);
        return err;
    }
    trans->reward = flt(num_int);
    addr++;

    // Done
    hb_mc_npa_t npa2 = {.x = re_mem_x, .y = re_mem_y, .epa = addr*4};
    err = hb_mc_manycore_read_mem(mc, &npa2, &num_int, sizeof(num_int));
    if (err != HB_MC_SUCCESS) {
        bsg_pr_err("%s: failed to read from manycore DMEM: %s\n", __func__, hb_mc_strerror(err));
        hb_mc_manycore_exit(mc);
        return err;
    }
    trans->done = num_int;

    return err;
}

void read_re_mem (hb_mc_manycore_t *mc, hb_mc_npa_t base_npa, uint32_t base_addr, int len) {
    hb_mc_idx_t re_mem_x = base_npa.x;
    hb_mc_idx_t re_mem_y = base_npa.y;
    hb_mc_epa_t re_mem_epa = base_npa.epa;
    uint32_t read_data;
    int err;
    for (size_t i = 0; i < len; i++) {
        hb_mc_npa_t npa = { .x = re_mem_x, .y = re_mem_y, .epa = (re_mem_epa + base_addr + i)*4 };
        err = hb_mc_manycore_read_mem(mc, &npa,
                &read_data, sizeof(read_data));
        if (err != HB_MC_SUCCESS) {
            bsg_pr_err("%s: failed to read A[%d] "
                   "from DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
                   i, re_mem_x, re_mem_y, base_addr + i);
        }
        printf("Read result(%d) %x \n", i, read_data);
    }
}

/*****************************************************************************************************************
* DRLP configure
******************************************************************************************************************/

void eva_offset_write_fp (hb_mc_manycore_t *mc, hb_mc_eva_t base_eva, uint32_t byte_offset, float number) {
    hb_mc_coordinate_t target = { .x = 1, .y = 1 };
    eva_t curr_eva = base_eva + (byte_offset<<2);
    uint32_t num_int = hex(number);
    hb_mc_manycore_eva_write(mc, &default_map, &target, &curr_eva, &num_int, sizeof(uint32_t));
}

void eva_offset_read_fp (hb_mc_manycore_t *mc, hb_mc_eva_t base_eva, uint32_t byte_offset, int len, float *read_float, bool print) {
    hb_mc_coordinate_t target = { .x = 1, .y = 1 };
    eva_t curr_eva = base_eva + (byte_offset<<2);
    uint32_t read_data[len];
    hb_mc_manycore_eva_read(mc, &default_map, &target, &curr_eva, read_data, len*sizeof(uint32_t));
    // translate to float point number
    uint32_t read_data_hex;
    for (size_t i = 0; i < len; i++) {
        read_data_hex = read_data[i];
        read_float[i] = flt(read_data_hex);
        if (print)
            printf("Read result[%d] = 0x%x(%f) \n", i, read_data_hex, read_float[i]);
    }
}

/*****************************************************************************************************************
* For DRLP Call DQN 
******************************************************************************************************************/

void fc_fp_drlp_map(NN_layer *fc) {
    int mod = (fc->input_size)%18;
    if (mod==0) {
        fc->slides = (fc->input_size)/18;
        fc->input_padding = 0;
    }
    else {
        fc->slides = (fc->input_size)/18 + 1;
        fc->input_padding = 18-mod;
    }
    fc->img_w_count = (fc->slides)*3;

    fc->ymove = fc->slides-1;
    
    if ((fc->output_size)>15) 
        fc->pe_on = 15;
    else
        fc->pe_on = fc->output_size-1;

    mod = (fc->output_size)%16;
    if (mod==0) {
        fc->zmove = (fc->output_size)/16;
        if (fc->zmove > 32) {
            bsg_pr_err("layer%d, output size is %d, zmove %d, Need to configure manully!\n", fc->layer, fc->output_size, fc->zmove);
        }
        fc->weight_padding = 0;
    }
    else {
        fc->zmove = (fc->output_size)/16 + 1;
        fc->weight_padding = 16-mod;
    }
}

void fc_dw_drlp_map(NN_layer *fc) {
    int quo = (fc->input_size)/18;
    int mod = (fc->input_size)%18;
    if (quo<16) {
        fc->zmove=1;
        fc->weight_padding = mod;
        if (mod == 0)
            fc->pe_on = quo-1;
        else
            fc->pe_on = quo;
    }
    else {
        fc->pe_on = 15;
        quo = (fc->input_size)/(18*16);
        mod = (fc->input_size)%(18*16);
        fc->weight_padding = mod;
        if (mod == 0)
            fc->zmove = quo;
        else
            fc->zmove = quo+1;
    }

    // output_size is actually dy size
    quo = (fc->output_size)/256;
    mod = (fc->output_size)%256;
    if (quo==0) {
        fc->xmove = 0;
        fc->ymove = mod-1;
        fc->input_padding = 0;
    }
    else {
        fc->ymove = 255;
        fc->input_padding = mod;
        if (mod==0) 
            fc->xmove = quo-1;
        else 
            fc->xmove = quo;
    }

    fc->img_w_count = (fc->xmove + 1)*(fc->ymove + 1);
}

void conv_fp_wrt_wgt (hb_mc_manycore_t *mc, hb_mc_eva_t drlp_dram_eva, NN_layer nn, float *weight, float *bias, uint32_t base_addr) {
    float number;
    uint32_t addr=base_addr;
    int index;
    
    if(nn.layer==0){
        for (int pe_split=0; pe_split<2; pe_split++) {
            int pe_offset = pe_split*16;
            // Bias
            for (int i=0; i<16; i++) {
                index = i+pe_offset;
                number = bias[index];
                eva_offset_write_fp(mc, drlp_dram_eva, addr, number);
                addr++;
            }
            // Weights
            for (int z=0; z<4; z++) {
                for (int pe=0; pe<16; pe++) {
                    for (int y=3; y>-1; y--) {
                        for (int x=0; x<4; x++) {
                            index = x + y*8 + z*8*8+ (pe+pe_offset)*8*8*4;
                            number = weight[index];
                            eva_offset_write_fp(mc, drlp_dram_eva, addr, number);
                            addr++;
                        }
                    }
                }
            }
            for (int z=0; z<4; z++) {
                for (int pe=0; pe<16; pe++) {
                    for (int y=3; y>-1; y--) {
                        for (int x=0; x<4; x++) {
                            index = x + (y+4)*8 + z*8*8+ (pe+pe_offset)*8*8*4;
                            number = weight[index];
                            eva_offset_write_fp(mc, drlp_dram_eva, addr, number);
                            addr++;
                        }
                    }
                }
            }
            for (int z=0; z<4; z++) {
                for (int pe=0; pe<16; pe++) {
                    for (int y=3; y>-1; y--) {
                        for (int x=0; x<4; x++) {
                            index = (x+4) + y*8 + z*8*8+ (pe+pe_offset)*8*8*4;
                            number = weight[index];
                            eva_offset_write_fp(mc, drlp_dram_eva, addr, number);
                            addr++;
                        }
                    }
                }
            }
            for (int z=0; z<4; z++) {
                for (int pe=0; pe<16; pe++) {
                    for (int y=3; y>-1; y--) {
                        for (int x=0; x<4; x++) {
                            index = (x+4) + (y+4)*8 + z*8*8+ (pe+pe_offset)*8*8*4;
                            number = weight[index];
                            eva_offset_write_fp(mc, drlp_dram_eva, addr, number);
                            addr++;
                        }
                    }
                }
            }
        }
    }
    else if(nn.layer==1) {
        for (int pe_split=0; pe_split<4; pe_split++) {
            int pe_offset = pe_split*16;
            // Bias
            for (int i=0; i<16; i++) {
                index = i+pe_offset;
                number = bias[index];
                eva_offset_write_fp(mc, drlp_dram_eva, addr, number);
                addr++;
            }
            // Weights
            for (int z=0; z<32; z++) {
                for (int pe=0; pe<16; pe++) {
                    for (int y=3; y>-1; y--) {
                        for (int x=0; x<4; x++) {
                            index = x + y*4 + z*4*4+ (pe+pe_offset)*4*4*32;
                            number = weight[index];
                            eva_offset_write_fp(mc, drlp_dram_eva, addr, number);
                            addr++;
                        }
                    }
                }
            }
        }
    }
    else {
        for (int pe_split=0; pe_split<4; pe_split++) {
            int pe_offset = pe_split*16;
            // Bias
            for (int i=0; i<16; i++) {
                index = i+pe_offset;
                number = bias[index];
                eva_offset_write_fp(mc, drlp_dram_eva, addr, number);
                addr++;
            }
            // Weights
            for (int z=0; z<32; z++) {
                for (int pe=0; pe<16; pe++) {
                    for (int y=2; y>-1; y--) {
                        for (int z_offset=0; z_offset<2; z_offset++) {
                            for (int x=0; x<3; x++) {
                                index = x + y*3 + (z*2+z_offset)*3*3+ (pe+pe_offset)*3*3*64;
                                number = weight[index];
                                eva_offset_write_fp(mc, drlp_dram_eva, addr, number);
                                addr++;
                            }
                        }
                    }
                }
            }
        }
    }
}


void fc_fp_wrt_wgt (hb_mc_manycore_t *mc, hb_mc_eva_t drlp_dram_eva, NN_layer fc, float *weight, float *bias, uint32_t base_addr) {
    fc_fp_drlp_map(&fc);

    float number;
    uint32_t addr=base_addr;
    int index;
    int input_padding=fc.input_padding;
    int slides=fc.slides, pe_on=fc.pe_on;
    int zmove=fc.zmove;
    int weight_size=fc.weight_size;
    for (int i=0; i<slides; i++) {
        for (int z=0; z<zmove; z++) {
            // Write bias
            if (i==0) { 
                for (int j = 0; j < (pe_on+1); j++) {
                    index = z+j*zmove;
                    if (index >= (fc.output_size))
                        number = 0.0;
                    else
                        number = bias[index];
                    eva_offset_write_fp(mc, drlp_dram_eva, addr, number);
                    addr++;
                    // printf("Write bias %f\n", number);
                }
            }
            else {
                if (z==0) {
                    number = 0.0;
                    eva_offset_write_fp(mc, drlp_dram_eva, addr, number);
                    addr++;
                }
            }
            // Write weight
            for (int j = 0; j < (pe_on+1); j++) {
                for (int k = 0; k < 18; k++) {
                    if (pe_on+1==16)
                        index = (18*i+k)*(fc.output_size)+z+j*zmove;
                    else
                        index = (18*i+k)*(pe_on+1)+z+j*zmove;
                    if (index >= weight_size)
                        number = 0.0;
                    else
                        number = weight[index];
                    eva_offset_write_fp(mc, drlp_dram_eva, addr, number);
                    addr++;
                    // if (fc.layer==3 && i==0 && z==0)
                        // printf("weight[%d] = %f \n", addr, number);
                }
            }
        }
    }
}


void conv_bp_wrt_wgt (hb_mc_manycore_t *mc, hb_mc_eva_t drlp_dram_eva, NN_layer nn, float *weight) {
    float number;
    int index;
    uint32_t addr = nn.wT_base_addr;
    
    if(nn.layer==0){
        // Action re-arrangement for conv1_dw
        // (1,84,84,4)->(64,4,23,18)
        float newA[64*4*23*18] = {0};
        for (int z=0; z<4; z++) {
            for (int x_base=0; x_base<8; x_base++) {
                for (int y_base=0; y_base<8; y_base++) {
                    int i = 0;
                    for (int x=x_base; x<x_base+80; x=x+4) {
                        for (int y=y_base; y<y_base+80; y=y+4) {
                            // bsg_pr_test_info("%d %d %d %d %d\n", z, x_base, y_base, x, y);
                            number = weight[x+y*84+z*84*84];
                            index = 8*x_base+y_base + z*64 + (i/18)*64*4 + (i%18)*64*4*23;
                            newA[index] = number;
                            i++;
                        }
                    }
                }
            }
        }

        for (int ff=0; ff<4; ff++) {
            for (int z=0; z<23; z++) {
                for (int x=0; x<1; x++) {
                    for (int y=0; y<64; y++) {
                        for (int i_base=2; i_base>-1; i_base--) {
                            for (int i=i_base*6; i<i_base*6+6; i++) {
                                index = y + (x+ff)*64 + z*64*4 + i*64*4*23;
                                number = newA[index];
                                eva_offset_write_fp(mc, drlp_dram_eva, addr, number);
                                addr++;
                            }
                        }
                    }
                }
            }
        }
    }
    else if(nn.layer==1) {
        // Weight re-arrangement for conv2 dx
        // Rotate W 180 degree and swicth dim2 and dim3, then split to 4
        for (int repeat=0; repeat<4; repeat++) {
            for (int pe_split=0; pe_split<2; pe_split++) {
                int pe_offset = pe_split*16;
                // Bias
                for (int i=0; i<16; i++) {
                    number = 0.0;
                    eva_offset_write_fp(mc, drlp_dram_eva, addr, number);
                    addr++;
                }
                // Weights
                for (int z_i=0; z_i<16; z_i++) {
                    for (int pe=0; pe<16; pe++) {
                        for (int y=1; y>-1; y--) {
                            for (int x=1; x>-1; x--) {
                                for (int z=z_i*4; z<(z_i*4+4); z++) {
                                    if (repeat == 0)
                                        index = (3- (x*2+1)) + (3- (y*2+1))*4 + (pe+pe_offset)*4*4+ z*4*4*32;
                                    if (repeat == 1)
                                        index = (3- (x*2)) + (3- (y*2+1))*4 + (pe+pe_offset)*4*4+ z*4*4*32;
                                    if (repeat == 2)
                                        index = (3- (x*2+1)) + (3- (y*2))*4 + (pe+pe_offset)*4*4+ z*4*4*32;
                                    if (repeat == 3)
                                        index = (3- (x*2)) + (3- (y*2))*4 + (pe+pe_offset)*4*4+ z*4*4*32;
                                    number = weight[index];
                                    eva_offset_write_fp(mc, drlp_dram_eva, addr, number);
                                    addr++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    else {
        // Weight re-arrangement for conv3 dx
        // Rotate W 180 degree and swicth dim2 and dim3 
        for (int pe_split=0; pe_split<4; pe_split++) {
            int pe_offset = pe_split*16;
            // Bias
            for (int i=0; i<16; i++) {
                number = 0.0;
                eva_offset_write_fp(mc, drlp_dram_eva, addr, number);
                addr++;
            }
            // Weights
            for (int z=0; z<32; z++) {
                for (int pe=0; pe<16; pe++) {
                    for (int y=2; y>-1; y--) {
                        for (int z_offset=0; z_offset<2; z_offset++) {
                            for (int x=0; x<3; x++) {
                                index = (2-x) + (2-y)*3 + (pe+pe_offset)*3*3+ (z*2+z_offset)*3*3*64;
                                number = weight[index];
                                eva_offset_write_fp(mc, drlp_dram_eva, addr, number);
                                addr++;
                            }
                        }
                    }
                }
            }
        }
    }
}

void fc_bp_wrt_wgt (hb_mc_manycore_t *mc, hb_mc_eva_t drlp_dram_eva, NN_layer nn, float *w) {
    float number;
    int index, index0, index1, index_T;
    uint32_t addr = nn.wT_base_addr;
    int total_row = nn.input_size;
    int total_col = nn.output_size;
    // host_transpose(w, wT, nn.input_size, nn.output_size);
    
    if(nn.layer==4) {
        for (int z=0; z<32; z++) {
            for (int j=0; j<16; j++) {
                number = 0.0;
                eva_offset_write_fp(mc, drlp_dram_eva, addr, number);
                addr++;
            }
            for (int j=0; j<16; j++) {
                for (int i=0; i<18; i++) {
                    if (i>=4) {
                        number = 0.0;
                    }
                    else {
                        index_T = z+j*32 + i*(nn.input_size);
                        index = (index_T%total_row)*total_col + (index_T/total_row);
                        number = w[index];
                    }
                    eva_offset_write_fp(mc, drlp_dram_eva, addr, number);
                    addr++;
                }
            }
        }
    }
    else {
        for (int repeat=0; repeat<7; repeat++) {
            for (int slides=0; slides<29; slides++) {
                for (int z=0; z<32; z++) {
                    if (slides==0) {
                        for (int j=0; j<16; j++) {
                            number = 0.0;
                            eva_offset_write_fp(mc, drlp_dram_eva, addr, number);
                            addr++;
                        }
                    }
                    else {
                        if (z==0) {
                            number == 0.0;
                            eva_offset_write_fp(mc, drlp_dram_eva, addr, number);
                            addr++;
                        }
                    }
                    for (int j=0; j<16; j++) {
                        for (int i=0; i<18; i++) {
                            index0 = slides*18+i;
                            index1 = (nn.output_size)*repeat + z + j*32;
                            if (index0 >= (nn.output_size) || index1>=(nn.input_size)) {
                                number = 0.0;
                            }
                            else {
                                index_T = index0*(nn.input_size) + index1;
                                index = (index_T%total_row)*total_col + (index_T/total_row);
                                number = w[index];
                            }
                            eva_offset_write_fp(mc, drlp_dram_eva, addr, number);
                            addr++;
                        }
                    }
                }
            }
        }
    }
}

// void wgt_transpose_and_write (hb_mc_manycore_t *mc, hb_mc_npa_t drlp_dram_npa, NN_layer fc, float *w, float *wT) {
//     int in_act_size = fc.input_size;
//     int out_act_size = fc.output_size;
//     host_transpose(w, wT, in_act_size, out_act_size);
//     fc.input_size = out_act_size;
//     fc.output_size = in_act_size;
//     float zero_bias[1000] = {0.0};
//     fc_fp_wrt_wgt(mc, drlp_dram_npa, fc, wT, zero_bias, fc.wT_base_addr);
// }
//
void read_fc_w (hb_mc_manycore_t *mc, hb_mc_eva_t drlp_dram_eva, NN_layer fc, float *weight, float *bias) {
    fc_fp_drlp_map(&fc);

    float number;
    uint32_t addr=fc.wgt_base_addr;
    int index;
    int input_padding=fc.input_padding;
    int slides=fc.slides, pe_on=fc.pe_on;
    int zmove=fc.zmove;
    int weight_size=fc.weight_size;
    for (int i=0; i<slides; i++) {
        for (int z=0; z<zmove; z++) {
            // Read bias
            if (i==0) { 
                for (int j = 0; j < (pe_on+1); j++) {
                    index = z+j*zmove;
                    if (index >= (fc.output_size))
                        number = 0.0;
                    else
                        eva_offset_read_fp(mc, drlp_dram_eva, addr, 1, &bias[index], false);
                    addr++;
                }
            }
            else {
                if (z==0) {
                    addr++;
                }
            }
            // Read weight
            for (int j = 0; j < (pe_on+1); j++) {
                for (int k = 0; k < 18; k++) {
                    if (pe_on+1==16)
                        index = (18*i+k)*(fc.output_size)+z+j*zmove;
                    else
                        index = (18*i+k)*(pe_on+1)+z+j*zmove;
                    if (index >= weight_size)
                        number = 0.0;
                    else
                        eva_offset_read_fp(mc, drlp_dram_eva, addr, 1, &weight[index], false);
                    addr++;
                }
            }
        }
    }
}



void read_fc_dw (hb_mc_manycore_t *mc, hb_mc_eva_t drlp_dram_eva, float *dW, NN_layer fc){
    int row = fc.input_size;
    int col = fc.output_size;
    if (row == 512)
        row = 540;
    if (row == 3136)
        row = 3168;
    uint32_t addr = fc.dw_base_addr+1;
    int index;
    for (int i = 0; i < col; i++) {
        for (int j = 0; j < row; j++) {
            if (i < fc.output_size && j < fc.input_size) {
                // index = i*row+j;
                index = i+j*col;
                eva_offset_read_fp(mc, drlp_dram_eva, addr, 1, &dW[index], false);
                // bsg_pr_test_info("Read dW[%d](%d) %.16f \n", index,  addr, dW[index]);
            }
            addr++;
        }
        if (row<18) 
            addr += (18-row);
    }
}

void read_fc_db (hb_mc_manycore_t *mc, float *db, NN_layer fc) {
    uint32_t dy_addr = fc.dy_base_addr;
    uint32_t y_addr = fc.rst_base_addr + 1;
    int size = fc.output_size;
    int read_data;
    for (int i = 0; i < size; i++) {
        hb_mc_npa_t npa2 = { .x = DRLP_X, .y = DRLP_Y, .epa = DRLP_RMEM_PREFIX + (dy_addr+i)*4 };
        hb_mc_manycore_read_mem(mc, &npa2, &read_data, sizeof(read_data));
        db[i] = flt(read_data);
        // if (i < 1000)
            bsg_pr_test_info("Read db(%d) %.4e from addr %d \n", i, db[i], dy_addr+i);
    }
}

void read_conv_dw (hb_mc_manycore_t *mc, hb_mc_eva_t drlp_dram_eva, float *dW, NN_layer nn){
    int layer = nn.layer;
    uint32_t addr = nn.dw_base_addr+1;
    int size = nn.weight_size;
    if (layer == 2) {
        for (int i = 0; i < 64; i++) {
            for (int j = 0; j < 64; j++) {
                for (int k = 0; k < 3; k++) {
                    for (int w = 0; w < 3; w++) {
                        int index = i*64*9 + j*9 + k + w*3;
                        eva_offset_read_fp(mc, drlp_dram_eva, addr, 1, &dW[index], false);
                        addr++;
                    }
                }
            }
        }
    }
    else if (layer == 1) {
        for (int i = 0; i < 64; i++) {
            for (int j = 0; j < 32; j++) {
                for (int k = 0; k < 4; k++) {
                    for (int w = 0; w < 4; w++) {
                        int index= i*32*16 + j*16 + k + w*4;
                        eva_offset_read_fp(mc, drlp_dram_eva, addr, 1, &dW[index], false);
                        addr++;
                    }
                }
            }
        }
    }
    else if (layer == 0) {
        for (int j = 0; j < 4; j++) {
            for (int i = 0; i < 32; i++) {
                for (int k = 0; k < 8; k++) {
                    for (int w = 0; w < 8; w++) {
                        int index= i*4*64 + j*64 + k + w*8;
                        eva_offset_read_fp(mc, drlp_dram_eva, addr, 1, &dW[index], false);
                        addr++;
                    }
                }
            }
        }
    }

}

void drlp_conv_fp(hb_mc_manycore_t *mc, NN_layer fc) {
    uint32_t config[DRLP_CFG_LEN] = {0, fc.act_base_addr, fc.wgt_base_addr, fc.rst_base_addr, 0, 0, 0};
    uint32_t done = 0;
    if (fc.layer==0) {
        done = 0;
        config[0] = 0x73D30413;
        config[5] = 0xDC800039;
        config[6] = 0x06002031;
        for (int k = 0; k < 8; ++k){
            write_configure(mc, config);
            // Wait for stop
            hb_mc_npa_t done_npa = { .x = DRLP_X, .y = DRLP_Y, .epa = DRLP_DONE_ADDR };
            while (done != 1) {
                for (int i=0; i<999; i++){}
                hb_mc_manycore_read_mem(mc, &done_npa, &done, sizeof(done));
            }
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
    }
    else if(fc.layer==1) {
        done = 0;
        config[0] = 0x6BC82008;
        config[5] = 0xB4000028;
        config[6] = 0x14008051;
        for (int k = 0; k < 4; ++k){
            write_configure(mc, config);
            // Wait for stop
            hb_mc_npa_t done_npa = { .x = DRLP_X, .y = DRLP_Y, .epa = DRLP_DONE_ADDR };
            while (done != 1) {
                for (int i=0; i<999; i++){}
                hb_mc_manycore_read_mem(mc, &done_npa, &done, sizeof(done));
            }
            done = 0;
            config[2] = config[2] + 8208;
            config[3] = config[3] + 1296;
            if (k==0) {
                config[5] = 0xB4000008;
            } 
        }
    }
    else if(fc.layer==2) {
        done = 0;
        config[0] = 0x27C62006;
        config[5] = 0x3F000028;
        config[6] = 0x24009051;
        for (int k = 0; k < 4; ++k){
            write_configure(mc, config);
            // Wait for stop
            hb_mc_npa_t done_npa = { .x = DRLP_X, .y = DRLP_Y, .epa = DRLP_DONE_ADDR };
            while (done != 1) {
                for (int i=0; i<999; i++){}
                hb_mc_manycore_read_mem(mc, &done_npa, &done, sizeof(done));
            }
            done = 0;
            config[2] = config[2] + 9232;
            config[3] = config[3] + 784;
            if (k==0) {
                config[5] = 0x3F000008;
            } 
        }
    }
} 

void drlp_fc_fp(hb_mc_manycore_t *mc, NN_layer fc) {
    fc_fp_drlp_map(&fc);

    int mode = 2, stride = 1;
    int relu = fc.relu, pe_on = fc.pe_on;    
    int xmove = 0, ymove = fc.ymove, zmove = fc.zmove;
    uint32_t config0 = (mode<<30) + (relu<<29) + (stride<<26) + (pe_on<<22)
        + (xmove<<16) + (zmove<<8) + ymove;
    
    int img_w_count = fc.img_w_count;
    int imem_r_base_addr=0, imem_update=1, imem_skip=0, wgt_mem_update=1;
    int bias_psum=0, wo_compute=0, wo_burst=0;
    uint32_t config5 = (img_w_count<<19) + (imem_r_base_addr<<6) + (imem_update<<5) + (imem_skip<<4)
        + (wgt_mem_update<<3)+ (bias_psum<<2) + (wo_compute<<1) + wo_burst;

    int layer, wgt_from_dram, img_from_dram, rst_to_dram;
    if (fc.input_src==DMA)
        img_from_dram=1;
    else
        img_from_dram=0;
    if (fc.output_dst==DMA)
        rst_to_dram=1;
    else
        rst_to_dram=0;
    layer = fc.layer;
    wgt_from_dram=1;
    uint32_t config6 = (layer<<28) + (wgt_from_dram<<26) + (img_from_dram<<25) + (rst_to_dram<<24) + 1;

    // bsg_pr_test_info("========DRLP FC FP========\n");
    // if (fc.layer == 3) {
        // config0 = 0xA7C020AE;
        // config5 = 0x10680028;
        // config6 = 0x34004C03;
    // }
    // else {
        // config0 = 0x84C0011C;
        // config5 = 0x02B80028;
        // config6 = 0x4400009B;
    // }
    uint32_t config[DRLP_CFG_LEN] = {config0, fc.act_base_addr, fc.wgt_base_addr, fc.rst_base_addr, 0, config5, config6};

    write_configure(mc, config);
    // Wait for stop
    uint32_t done = 0;
    hb_mc_npa_t done_npa = { .x = DRLP_X, .y = DRLP_Y, .epa = DRLP_DONE_ADDR };
    while (done != 1) {
        for (int i=0; i<999; i++){}
        hb_mc_manycore_read_mem(mc, &done_npa, &done, sizeof(done));
    }
} 


void drlp_fc_dw(hb_mc_manycore_t *mc, NN_layer fc) {
    fc_dw_drlp_map(&fc);

    int mode = 3, stride = 1, relu = 0;
    int pe_on = fc.pe_on;
    int xmove = fc.xmove, ymove = fc.ymove, zmove = fc.zmove;
    uint32_t config0 = (mode<<30) + (relu<<29) + (stride<<26) + (pe_on<<22) + (xmove<<16) + (zmove<<8) + ymove;
    
    int img_w_count = fc.img_w_count;
    int imem_r_base_addr=0, imem_update=1, imem_skip=0, wgt_mem_update=1;
    int bias_psum=1, wo_compute=0, wo_burst=0;
    uint32_t config5 = (img_w_count<<19) + (imem_r_base_addr<<6) + (imem_update<<5) + (imem_skip<<4) + (wgt_mem_update<<3)+ (bias_psum<<2) + (wo_compute<<1) + wo_burst;

    int layer, wgt_from_dram, img_from_dram, rst_to_dram;
    uint32_t config2;
    if (fc.layer==4) {
        img_from_dram = 1;
        layer = 5;
        wgt_from_dram=0;
        config2 = fc.act_base_addr+1;
    }
    else {
        img_from_dram=0;
        layer=7;
        wgt_from_dram=1; 
        config2 = fc.act_base_addr-1; // first layer
    }
    rst_to_dram=1;
    uint32_t config6 = (layer<<28) + (wgt_from_dram<<26) + (img_from_dram<<25) + (rst_to_dram<<24) + 1;

    // bsg_pr_test_info("========DRLP dW========\n");
    uint32_t config[DRLP_CFG_LEN] = {config0, fc.dy_base_addr, config2, fc.dw_base_addr, 0, config5, config6};
    write_configure(mc, config);
    // Wait for stop
    uint32_t done = 0;
    hb_mc_npa_t done_npa = { .x = DRLP_X, .y = DRLP_Y, .epa = DRLP_DONE_ADDR };
    while (done != 1) {
        for (int i=0; i<999; i++){}
        hb_mc_manycore_read_mem(mc, &done_npa, &done, sizeof(done));
    }
} 

void drlp_fc_dx(hb_mc_manycore_t *mc, NN_layer fc) {
    // Switch in and out for back prop
    int in_act_size = fc.input_size;
    int out_act_size = fc.output_size;
    fc.input_size = out_act_size;
    fc.output_size = in_act_size;
    fc_fp_drlp_map(&fc);

    int mode = 2, stride = 1;
    int relu = fc.relu, pe_on = fc.pe_on;    
    int xmove = 0, ymove = fc.ymove, zmove = fc.zmove;
    uint32_t config0 = (mode<<30) + (relu<<29) + (stride<<26) + (pe_on<<22)
        + (xmove<<16) + (zmove<<8) + ymove;
    
    int img_w_count = fc.img_w_count;
    int imem_r_base_addr=0, imem_update=1, imem_skip=0, wgt_mem_update=1;
    int bias_psum=0, wo_compute=0, wo_burst=0;
    uint32_t config5 = (img_w_count<<19) + (imem_r_base_addr<<6) + (imem_update<<5) + (imem_skip<<4)
        + (wgt_mem_update<<3)+ (bias_psum<<2) + (wo_compute<<1) + wo_burst;

    int layer, wgt_from_dram, img_from_dram, rst_to_dram;
    if (fc.layer==4) {
        img_from_dram = 1;
        layer = 6;
    }
    else {
        img_from_dram=0;
        layer=8;
    }
    rst_to_dram=0;
    wgt_from_dram=1;
    uint32_t config6 = (layer<<28) + (wgt_from_dram<<26) + (img_from_dram<<25) + (rst_to_dram<<24) + 1;

    // bsg_pr_test_info("========DRLP dX========\n");
    uint32_t config[DRLP_CFG_LEN] = {config0, fc.dy_base_addr, fc.wT_base_addr, fc.dx_base_addr, fc.act_base_addr+1, config5, config6};
    write_configure(mc, config);
    // Wait for stop
    uint32_t done = 0;
    hb_mc_npa_t done_npa = { .x = DRLP_X, .y = DRLP_Y, .epa = DRLP_DONE_ADDR };
    while (done != 1) {
        for (int i=0; i<999; i++){}
        hb_mc_manycore_read_mem(mc, &done_npa, &done, sizeof(done));
    }
} 

void nn_fp(hb_mc_manycore_t *mc, hb_mc_eva_t drlp_dram_eva, float *state, NN_layer *nn, int num_layer, float* results) {
    bsg_pr_test_info("========Write state to DRAM on device========\n");
    float number;
    uint32_t addr=STATE_ADDR;
    for (int z = 0; z < 4; z++) { 
        for (int x = 0; x < 21; x++) { 
            for (int y = 0; y < 84; y++) { 
                for (int i = 0; i < 4; i++) { 
                    number = state[x*4+i + y*84 + z*84*84];
                    eva_offset_write_fp(mc, drlp_dram_eva, addr, number);
                    addr++; 
                }
            }
        }
    }

    bsg_pr_test_info("========Call DRLP NN FP========\n");
    for (int i = 0; i < num_layer; i++) { 
        bsg_pr_test_info("FP layer%d\n",i);
        if(nn[i].FC_CONV) {
            // CONV
            drlp_conv_fp(mc, nn[i]);
        }
        else {
            // FC
            drlp_fc_fp(mc, nn[i]);
        }
    }

    // bsg_pr_test_info("========Read FP results========\n");
    eva_offset_read_fp(mc, drlp_dram_eva, FP_RST_ADDR+1, ACTION_SIZE, results, false);
}

void nn_bp(hb_mc_manycore_t *mc, hb_mc_eva_t drlp_dram_eva, float *dy, NN_layer *nn, int num_layer) {
    bsg_pr_test_info("========Write state to DRAM on device========\n");
    int end=18;
    if (ACTION_SIZE>18)
        end = STATE_SIZE;
    float number;
    uint32_t addr;
    for (int i = 0; i < end; i++) { 
        if (i<ACTION_SIZE)
            number = dy[i];
        else
            number = 0.0;
        addr = OUT_GD_ADDR+ i; 
        eva_offset_write_fp(mc, drlp_dram_eva, addr, number);
    }

    bsg_pr_test_info("========Call DRLP NN BP========\n");
    // drlp_fc_dw(mc, nn[4]);
    fc2_dw(mc);
    bsg_pr_test_info("========1========\n");
    for (int i=0; i<9999; i++){}
    fc2_dx(mc);
    // drlp_fc_dx(mc, nn[4]);
    bsg_pr_test_info("========2========\n");
    fc1_dw(mc);
    // drlp_fc_dw(mc, nn[3]);
    bsg_pr_test_info("========3========\n");
    for (int i=0; i<9999; i++){}
    fc1_dx(mc);
    // drlp_fc_dx(mc, nn[3]);
    bsg_pr_test_info("========4========\n");

    conv3_dw(mc);
    bsg_pr_test_info("========5========\n");
    conv3_dx(mc);
    bsg_pr_test_info("========6====\n");
    conv2_dw(mc);
    bsg_pr_test_info("========7====\n");
    conv2_dx(mc);
    bsg_pr_test_info("========8====\n");
    conv1_dw(mc);
    bsg_pr_test_info("========9====\n");
}

/*****************************************************************************************************************
* High-level DQN API
******************************************************************************************************************/

void dqn_act(hb_mc_manycore_t *mc, hb_mc_eva_t drlp_dram_eva, Transition *trans, NN_layer *nn, int num_layer, float epsilon, float *results) {
    float prob = rand()/(float)(RAND_MAX);
    if (prob<epsilon) {
        trans->action = rand()%ACTION_SIZE;
    }
    else {
        nn_fp(mc, drlp_dram_eva, trans->state, nn, num_layer, results);
        int max_index = 0;
        for (int i = 0; i < ACTION_SIZE; i++) { 
            if (results[i] > results[max_index])
                max_index = i;
        }
        trans->action = max_index;
    }
}

void dqn_train(hb_mc_manycore_t *mc,  hb_mc_eva_t drlp_dram_eva, Transition *trans, NN_layer *nn, int num_layer, float* fc2_dy, float gamma) {
    // FP
    // next state
    float next_values[ACTION_SIZE];
    int next_max_index = 0;
    nn_fp(mc, drlp_dram_eva, trans->next_state, nn, num_layer, next_values);

    for (int i = 0; i < ACTION_SIZE; i++) { 
        if (next_values[i] > next_values[next_max_index])
            next_max_index = i;
        bsg_pr_test_info("DRLP Train: next_value[%d]=%f\n", i, next_values[i]);
    }
    // state
    float state_values[ACTION_SIZE];
    nn_fp(mc, drlp_dram_eva, trans->state, nn, num_layer, state_values);
    for (int i = 0; i < ACTION_SIZE; i++) { 
        bsg_pr_test_info("DRLP Train: state_value[%d]=%f\n", i, state_values[i]);
    }

    // Loss function
    float target;
    uint32_t action = trans->action;
    bsg_pr_test_info("DRLP Train: reward=%f\n", trans->reward);
    bsg_pr_test_info("DRLP Train: done=%f\n", trans->done);
    bsg_pr_test_info("DRLP Train: action=%d\n", action);
    if ((trans->done) == 0.0)
        target = (trans->reward) + gamma*next_values[next_max_index];
    else
        target = trans->reward;
    bsg_pr_test_info("DRLP Train: target=%f\n", target);

    for (int i = 0; i < ACTION_SIZE; i++) { 
        fc2_dy[i] = 0;
    }
    fc2_dy[action] = 2*(state_values[action] - target); // MSE loss function
    // bsg_pr_test_info("DRLP Train: reward=%f\n", trans->reward);
    for (int i = 0; i < ACTION_SIZE; i++) { 
        bsg_pr_test_info("DRLP Train: fc2_dy[%d]=%f\n", i, fc2_dy[i]);
    }

    // BP
    nn_bp(mc, drlp_dram_eva, fc2_dy, nn, num_layer);

}
