#include <Python.h>
#include <numpy/arrayobject.h>
#include "test_drlp_fpbp.h"
#include "test_drlp_host_gd.h"
#include <math.h>

#define hex(X) (*(int*)&X)
#define flt(X) (*(float*)&X)

// Test game
#define STATE_SIZE 4
#define ACTION_SIZE 2

// NN config
#define FC1_Y_SIZE 144
#define FC1_W_SIZE STATE_SIZE*FC1_Y_SIZE
#define FC2_Y_SIZE ACTION_SIZE
#define FC2_W_SIZE FC1_Y_SIZE*FC2_Y_SIZE
// In DRLP
#define FC1_Y_ADDR 0
#define FC2_dX_ADDR FC1_Y_ADDR+FC1_Y_SIZE+100
// DRAM
#define STATE_ADDR 1
#define FC1_W_ADDR STATE_ADDR+18+(100)
// #define FC2_W_ADDR FC1_W_ADDR+FC1_W_SIZE+(100)
#define FC2_W_ADDR FC1_W_ADDR+((18*FC1_Y_SIZE)*16)
#define FC2_WT_ADDR FC2_W_ADDR+(FC2_W_SIZE*16)
#define FC2_Y_ADDR FC2_WT_ADDR+(FC2_W_SIZE*16)
#define FC2_dY_ADDR FC2_Y_ADDR+(FC2_Y_SIZE*16)

#define FC2_dW_ADDR FC2_dY_ADDR+(18*16)
#define FC1_dW_ADDR FC2_dW_ADDR+(FC2_W_SIZE*16)

#define LR 0.001

// RL
#define RE_MEM_SIZE 5000
#define RE_MEM_INIT_SIZE 1
#define TRANSITION_SIZE 2*STATE_SIZE+3
#define STEP_MAX 3
#define TRAIN_FREQ 1

// Manycore
#define DRLP_X 3 
#define DRLP_Y 4
#define DRAM_X 3
#define DRAM_Y 5
#define RE_DRAM_X 2
#define RE_DRAM_Y 5

// Others
#define ONCHIP 0
#define DMA 1
#define HOST_COMPARE true

typedef struct {
	float state[STATE_SIZE];
	float next_state[STATE_SIZE];
	float reward;
	uint32_t done;
	uint32_t action;
} Transition;

typedef struct {
	int input_size;
	int output_size;
	int weight_size;
	bool input_src; // 0: on-chip, 1: DRAM
	bool output_dst;
	bool relu;
	int layer; // layer=4 means last fc layer, otherwise = 3
	// for drlp mapping
	int input_padding;
	int weight_padding;
	int slides;
	uint32_t act_base_addr;
	uint32_t wgt_base_addr;
	uint32_t rst_base_addr;
	uint32_t dy_base_addr;
	uint32_t dw_base_addr;
	uint32_t wT_base_addr;
	uint32_t dx_base_addr;
	int pe_on;
	int ymove;
	int zmove;
	int xmove;
	int img_w_count;
} FC_layer;

void param_random(float *param, int size){
	for (int i = 0; i < size; i++)
		param[i] = rand()/(float)(RAND_MAX);
}

/*****************************************************************************************************************
* For Python Calling 
******************************************************************************************************************/

PyObject* py_init(char *game_name) {
    Py_Initialize(); 
	import_array();

	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('../../regression/drlp_test/')"); // Add regression directory to PATH
	/* PyRun_SimpleString("print(sys.path)\n"); */
      
	PyObject *pmod, *pclass, *pargs, *pinst;

	// Load module
	pmod   = PyImport_ImportModule("gym_env");
	if (pmod == NULL)
		bsg_pr_err("Can't load module");
	pclass = PyObject_GetAttrString(pmod, "gym_env");
	Py_DECREF(pmod);

	// Load class
	pargs  = Py_BuildValue("(s)", game_name);
	pinst  = PyEval_CallObject(pclass, pargs);
	Py_DECREF(pclass);
	Py_DECREF(pargs);

	return pinst;
}

void call_reset(Transition *trans, PyObject *pinst) { 
	PyObject *preset = PyObject_GetAttrString(pinst, "reset");
	PyObject *pargs  = Py_BuildValue("()");
	PyArrayObject *pstate = PyEval_CallObject(preset, pargs);
	Py_DECREF(pargs);
	Py_DECREF(preset);
	if (PyArray_Check(pstate)) {
		int rows = pstate->dimensions[0];
		int row_step = pstate->strides[0];
		if (rows == STATE_SIZE) {
			for (int r=0; r<rows; r++){
				trans->state[r] = *(double*)(pstate->data + r*row_step);
			}
		}
		else {
			printf("Returned state has wrong size %d!\n", rows);
		}
	}
	else {
		printf("Returned state is not Numpy array!\n");
	}
	Py_DECREF(pstate);
	trans->done = 0;
}

void call_step(Transition *trans, PyObject *pinst) { 
	PyObject *pstep  = PyObject_GetAttrString(pinst, "step");
	PyObject *pargs  = Py_BuildValue("(i)", trans->action);
	PyArrayObject *pcatall = PyEval_CallObject(pstep, pargs);
	Py_DECREF(pargs);
	Py_DECREF(pstep);
	if (PyArray_Check(pcatall)) {
		int rows = pcatall->dimensions[0];
		int row_step = pcatall->strides[0];
		// Read next state
		for (int r=0; r<rows-2; r++){
			trans->next_state[r] = *(double*)(pcatall->data + r*row_step);
			// printf("state[%d,] is % f\n", r, trans->next_state[r]);
		}
		// Read reward and done
		trans->reward = *(double*)(pcatall->data + (rows-2)*row_step);
		// printf("Inside Reward is % f\n", trans->reward);
		trans->done = *(int*)(pcatall->data + (rows-1)*row_step);
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

uint32_t re_mem_push(hb_mc_manycore_t *mc, hb_mc_npa_t base_npa, Transition *trans, uint32_t position) {
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
* For DRLP Call DQN 
******************************************************************************************************************/

void fc_fp_drlp_map(FC_layer *fc) {
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
		if (fc->zmove > 32)
			bsg_pr_err("Need to configure manully!\n");
		fc->weight_padding = 0;
	}
	else {
		fc->zmove = (fc->output_size)/16 + 1;
		fc->weight_padding = 16-mod;
	}
}

void fc_dw_drlp_map(FC_layer *fc) {
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

void fc_fp_wrt_wgt (hb_mc_manycore_t *mc, FC_layer fc, float *weight, float *bias, uint32_t base_addr) {
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
					write_dram_float(mc, addr, number);
					addr++;
					// printf("Write bias %f\n", number);
				}
			}
			else {
				if (z==0) {
					number = 0.0;
					write_dram_float(mc, addr, number);
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
					// printf("weight[%d] = %f \n", index, number);
					write_dram_float(mc, addr, number);
					addr++;
				}
			}
		}
	}
}

void wgt_transpose_and_write (hb_mc_manycore_t *mc, FC_layer fc, float *w, float *wT) {
	int in_act_size = fc.input_size;
	int out_act_size = fc.output_size;
	host_transpose(w, wT, in_act_size, out_act_size);
	fc.input_size = out_act_size;
	fc.output_size = in_act_size;
	float zero_bias[1000] = {0.0};
	fc_fp_wrt_wgt(mc, fc, wT, zero_bias, fc.wT_base_addr);
}

void read_dw (hb_mc_manycore_t *mc, float *dW, FC_layer fc){
    int row = fc.input_size;
    int col = fc.output_size;
	uint32_t addr = fc.dw_base_addr+1;
	int index;
	int read_data;
	for (int i = 0; i < col; i++) {
		for (int j = 0; j < row; j++) {
			hb_mc_npa_t npa = { .x = dram_coord_x, .y = dram_coord_y , .epa = addr*4 };
			hb_mc_manycore_read_mem(mc, &npa,
					      &read_data, sizeof(read_data));
			index = i*row+j;
			dW[index] = flt(read_data);
			addr++;
            // bsg_pr_test_info("Read dW(%d) (%x)%.16f \n", addr, read_data, dW[index]);
			/* bsg_pr_test_info("dY_host %.5f, X_host %.5f \n", dY_host[i], X_host[j]); */
		}
		if (row<18) 
			addr += (18-row);
	}
}

void read_db (hb_mc_manycore_t *mc, float *db, FC_layer fc) {
	uint32_t dy_addr = fc.dy_base_addr;
	uint32_t y_addr = fc.rst_base_addr + 1;
    int size = fc.output_size;
	int read_data;
	for (int i = 0; i < size; i++) {
        hb_mc_npa_t npa = { .x = drlp_coord_x, .y = drlp_coord_y, .epa = DRLP_RMEM_PREFIX + (y_addr+i)*4 };
		hb_mc_manycore_read_mem(mc, &npa, &read_data, sizeof(read_data));
        // bsg_pr_test_info("Read y(%d) %x \n", i, read_data);
        if (read_data != 0) {
            hb_mc_npa_t npa2 = { .x = drlp_coord_x, .y = drlp_coord_y, .epa = DRLP_RMEM_PREFIX + (dy_addr+i)*4 };
		    hb_mc_manycore_read_mem(mc, &npa2, &read_data, sizeof(read_data));
            db[i] = flt(read_data);
        }
        else {
            db[i] = 0.0;
        }
        // bsg_pr_test_info("Read db(%d) %.5f \n", i, db[i]);
    }
}


void drlp_fc_fp(hb_mc_manycore_t *mc, FC_layer fc) {
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

void drlp_fc_dw(hb_mc_manycore_t *mc, FC_layer fc) {
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

void drlp_fc_dx(hb_mc_manycore_t *mc, FC_layer fc) {
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

void nn_fp(hb_mc_manycore_t *mc, float *state, FC_layer *nn, int num_layer, float* results) {
    // bsg_pr_test_info("========Write state to DRAM on device========\n");
	int end=18;
	if (STATE_SIZE>18)
		end = STATE_SIZE;
	float number;
	uint32_t addr;
	for (int i = 0; i < end; i++) { 
		if (i<STATE_SIZE)
			number = state[i];
		else
			number = 0.0;
		addr = STATE_ADDR + i; 
		write_dram_float(mc, addr, number);
	}

    bsg_pr_test_info("========Call DRLP NN FP========\n");
	for (int i = 0; i < num_layer; i++) { 
		drlp_fc_fp(mc, nn[i]);
	}

    // bsg_pr_test_info("========Read FP results========\n");
	read_dram(mc, FC2_Y_ADDR+1, ACTION_SIZE, results, false);
}

void nn_bp(hb_mc_manycore_t *mc, float *dy, FC_layer *nn, int num_layer) {
	// bsg_pr_test_info("========Write state to DRAM on device========\n");
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
		addr = FC2_dY_ADDR + i; 
		write_dram_float(mc, addr, number);
	}

    bsg_pr_test_info("========Call DRLP NN BP========\n");
	for (int i = num_layer-1; i > 0; i--) { 
		drlp_fc_dw(mc, nn[i]);
		// delay
		for (int i=0; i<9999; i++){}
		drlp_fc_dx(mc, nn[i]);
	}
		drlp_fc_dw(mc, nn[0]);
}

/*****************************************************************************************************************
* High-level DQN API
******************************************************************************************************************/

void dqn_act(hb_mc_manycore_t *mc, Transition *trans, FC_layer *nn, int num_layer, float epsilon) {
	float number;
	int addr;
	float prob = rand()/(float)(RAND_MAX);
	if (prob<epsilon) {
		trans->action = rand()%ACTION_SIZE;
	}
	else {
		float results[ACTION_SIZE];
		nn_fp(mc, trans->state, nn, num_layer, results);
		int max_index = 0;
		for (int i = 1; i < ACTION_SIZE; i++) { 
			if (results[i] > results[max_index])
				max_index = i;
		}
		trans->action = max_index;
	}
}

void dqn_train(hb_mc_manycore_t *mc, Transition *trans, FC_layer *nn, int num_layer, float* fc2_dy, float gamma) {
	// FP
	// next state
    for (int i = 0; i < STATE_SIZE; i++)
        bsg_pr_test_info("DRLP Train: next_state[%d]=%f\n", i, trans->next_state[i]);
	float next_values[ACTION_SIZE];
	int next_max_index = 0;
	nn_fp(mc, trans->next_state, nn, num_layer, next_values);

	for (int i = 0; i < ACTION_SIZE; i++) { 
		if (next_values[i] > next_values[next_max_index])
			next_max_index = i;
        bsg_pr_test_info("DRLP Train: next_value[%d]=%f\n", i, next_values[i]);
	}
	// state
    for (int i = 0; i < STATE_SIZE; i++)
        bsg_pr_test_info("DRLP Train: state[%d]=%f\n", i, trans->state[i]);
	float state_values[ACTION_SIZE];
	nn_fp(mc, trans->state, nn, num_layer, state_values);
    for (int i = 0; i < ACTION_SIZE; i++) { 
        bsg_pr_test_info("DRLP Train: state_value[%d]=%f\n", i, state_values[i]);
    }

	// Loss function
	float target = (trans->reward) + gamma*next_values[next_max_index];
	// float fc2_dy[ACTION_SIZE]={0.0};
	fc2_dy[next_max_index] = target - state_values[next_max_index]; // MSE loss function
    bsg_pr_test_info("DRLP Train: reward=%f\n", trans->reward);
    for (int i = 0; i < ACTION_SIZE; i++) { 
        bsg_pr_test_info("DRLP Train: fc2_dy[%d]=%f\n", i, fc2_dy[i]);
    }

	// BP
	nn_bp(mc, fc2_dy, nn, num_layer);

	// float drlp_fc2_dw[288];
	// read_dram(mc, FC2_dW_ADDR+1, 4, drlp_fc2_dw);
	// float drlp_fc1_dw[20];
	// read_dram(mc, FC1_dW_ADDR+1, 20, drlp_fc2_dw);
	

}
