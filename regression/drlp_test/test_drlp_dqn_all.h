#include <Python.h>
#include <numpy/arrayobject.h>
#include "test_drlp_fpbp.h"
#include "test_drlp_host_gd.h"
#include <math.h>

#define hex(X) (*(int*)&X)
#define flt(X) (*(float*)&X)

// Test game
#define FRAME_SIZE (84*84)
#define FRAME_NUM 4 
#define STATE_SIZE FRAME_SIZE*FRAME_NUM 
#define ACTION_SIZE 4

// NN config
#define CONV1_W_SIZE (8*8*4*32)
#define CONV1_Y_SIZE (20*20*32)
#define CONV2_W_SIZE (4*4*32*64)
#define CONV2_Y_SIZE (9*9*64)
#define CONV3_W_SIZE (3*3*64*64)
#define CONV3_Y_SIZE (7*7*64)
// #define FC1_Y_SIZE 512
#define FC1_Y_SIZE 18
#define FC1_W_SIZE CONV3_Y_SIZE*FC1_Y_SIZE
#define FC2_Y_SIZE ACTION_SIZE
#define FC2_W_SIZE FC1_Y_SIZE*FC2_Y_SIZE
// In DRLP
#define RMEM_ADDR0 0 
#define RMEM_ADDR1 12800
#define RMEM_ADDR2 17984 
#define RMEM_ADDR3 21120 
#define RMEM_ADDR4 21632 
// DRAM
#define STATE_ADDR 0
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
#define FP_RST_ADDR (5500*1024)

// RL
#define RE_MEM_SIZE 5000
#define RE_MEM_INIT_SIZE 1
#define TRANSITION_SIZE 2*STATE_SIZE+3
#define STEP_MAX 1
#define TRAIN_FREQ 1
#define LR 0.001

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
#define HOST_COMPARE false

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
	int layer; // layer=4 means last fc layer, 3 mean other FC
    bool FC_CONV;
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
} NN_layer;

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
    bsg_pr_test_info("Reset\n");
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
        // bsg_pr_test_info("chas:%d, cha_step:%d, rows:%d, row_step:%d, cols:%d, col_step:%d\n", chas, cha_step, rows, row_step, cols, col_step);
	    for (int d=0; d<chas; d++){
	        for (int r=0; r<rows; r++){
		    	for (int c=0; c<cols; c++){
		    		trans->state[d*(rows*cols)+r*cols+c] = *(float*)(pstate->data + d*cha_step + r*row_step + c*col_step);
                    // printf("state[%d,] is % f\n", r, trans->state[r]);
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
		trans->done = *(int*)(pcatall->data + (rows-1)*row_step + 1*col_step);
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

uint32_t re_mem_push(hb_mc_manycore_t *mc, Transition *trans, uint32_t position) {
	int rc;
	float number;
	uint32_t num_int;
	uint32_t trans_size = TRANSITION_SIZE;
	uint32_t addr = (position*trans_size);
	// State
    bsg_pr_test_info("push state\n");
	for (int i=0; i<STATE_SIZE; i++) {
		number = trans->state[i];
        // bsg_pr_test_info("state[%d]=%f\n",i,number);
		num_int = hex(number);
		hb_mc_npa_t npa = {.x = RE_DRAM_X, .y = RE_DRAM_Y, .epa = addr*4};
		rc = hb_mc_manycore_write_mem(mc, &npa, &num_int, sizeof(num_int));
		if (rc != HB_MC_SUCCESS) {
			bsg_pr_err("%s: failed to write state 0x%08" PRIx32 " "
				   "to DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
				   __func__, num_int, RE_DRAM_X, RE_DRAM_Y, addr);
			hb_mc_manycore_exit(mc);
		}
		addr++;
	}

	// Next state
    bsg_pr_test_info("push next state\n");
	for (int i=0; i<STATE_SIZE; i++) {
		number = trans->next_state[i];
		num_int = hex(number);
		hb_mc_npa_t npa = {.x = RE_DRAM_X, .y = RE_DRAM_Y, .epa = addr*4};
		rc = hb_mc_manycore_write_mem(mc, &npa, &num_int, sizeof(num_int));
		if (rc != HB_MC_SUCCESS) {
			bsg_pr_err("%s: failed to write next_state 0x%08" PRIx32 " "
				   "to DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
				   __func__, num_int, RE_DRAM_X, RE_DRAM_Y, addr);
			hb_mc_manycore_exit(mc);
		}
		addr++;
	}

	// Action
    bsg_pr_test_info("push next action\n");
	num_int = trans->action;
	hb_mc_npa_t npa0 = {.x = RE_DRAM_X, .y = RE_DRAM_Y, .epa = addr*4};
	rc = hb_mc_manycore_write_mem(mc, &npa0, &num_int, sizeof(num_int));
	if (rc != HB_MC_SUCCESS) {
		bsg_pr_err("%s: failed to write action 0x%08" PRIx32 " "
			   "to DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
			   __func__, num_int, RE_DRAM_X, RE_DRAM_Y, addr);
		hb_mc_manycore_exit(mc);
	}
	addr++;

	// Reward
    bsg_pr_test_info("push reward\n");
	num_int = hex(trans->reward);
	hb_mc_npa_t npa1 = {.x = RE_DRAM_X, .y = RE_DRAM_Y, .epa = addr*4};
	rc = hb_mc_manycore_write_mem(mc, &npa1, &num_int, sizeof(num_int));
	if (rc != HB_MC_SUCCESS) {
		bsg_pr_err("%s: failed to write reward 0x%08" PRIx32 " "
			   "to DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
			   __func__, num_int, RE_DRAM_X, RE_DRAM_Y, addr);
		hb_mc_manycore_exit(mc);
	}
	addr++;

	// Done
    bsg_pr_test_info("push done\n");
	num_int = trans->done;
	hb_mc_npa_t npa2 = {.x = RE_DRAM_X, .y = RE_DRAM_Y, .epa = addr*4};
	rc = hb_mc_manycore_write_mem(mc, &npa2, &num_int, sizeof(num_int));
	if (rc != HB_MC_SUCCESS) {
		bsg_pr_err("%s: failed to write done 0x%08" PRIx32 " "
			   "to DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
			   __func__, num_int, RE_DRAM_X, RE_DRAM_Y, addr);
		hb_mc_manycore_exit(mc);
	}

	if ((position+1)==RE_MEM_SIZE)
		return 0;
	else
		return position+1;
}

int re_mem_sample(hb_mc_manycore_t *mc, Transition *trans, uint32_t size) {
	// uint32_t position = (rand()%size);
	uint32_t position = 0;
	uint32_t trans_size = TRANSITION_SIZE;
	uint32_t addr = position*trans_size;
	uint32_t num_int;
	int err;
    printf("Sample %dth transition from replay memory addr%d.\n", position, addr);

	// State
    bsg_pr_test_info("sample state\n");
	for (int i=0; i<STATE_SIZE; i++) {
		hb_mc_npa_t npa = { .x = RE_DRAM_X, .y = RE_DRAM_Y, .epa = addr*4 };
		err = hb_mc_manycore_read_mem(mc, &npa, &num_int, sizeof(num_int));
		trans->state[i] = flt(num_int);
        // if (i<100 || i > (STATE_SIZE-100))
            // printf("state[%d]=%f.\n", i, trans->state[i]);

		if (err != HB_MC_SUCCESS) {
			bsg_pr_err("%s: failed to read from replay memory: %s\n", __func__, hb_mc_strerror(err));
			hb_mc_manycore_exit(mc);
			return err;
		}
		addr++;
	}

	// Next state
    bsg_pr_test_info("sample next state\n");
	for (int i=0; i<STATE_SIZE; i++) {
		hb_mc_npa_t npa = { .x = RE_DRAM_X, .y = RE_DRAM_Y, .epa = addr*4 };
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
    bsg_pr_test_info("sample action\n");
	hb_mc_npa_t npa0 = {.x = RE_DRAM_X, .y = RE_DRAM_Y, .epa = addr*4};
	err = hb_mc_manycore_read_mem(mc, &npa0, &num_int, sizeof(num_int));
	if (err != HB_MC_SUCCESS) {
		bsg_pr_err("%s: failed to read from replay memory: %s\n", __func__, hb_mc_strerror(err));
		hb_mc_manycore_exit(mc);
		return err;
	}
	trans->action = num_int;
	addr++;

	// Reward
    bsg_pr_test_info("sample reward\n");
	hb_mc_npa_t npa1 = {.x = RE_DRAM_X, .y = RE_DRAM_Y, .epa = addr*4};
	err = hb_mc_manycore_read_mem(mc, &npa1, &num_int, sizeof(num_int));
	if (err != HB_MC_SUCCESS) {
		bsg_pr_err("%s: failed to read from manycore DMEM: %s\n", __func__, hb_mc_strerror(err));
		hb_mc_manycore_exit(mc);
		return err;
	}
	trans->reward = flt(num_int);
	addr++;

	// Done
    bsg_pr_test_info("sample done\n");
	hb_mc_npa_t npa2 = {.x = RE_DRAM_X, .y = RE_DRAM_Y, .epa = addr*4};
	err = hb_mc_manycore_read_mem(mc, &npa2, &num_int, sizeof(num_int));
	if (err != HB_MC_SUCCESS) {
		bsg_pr_err("%s: failed to read from manycore DMEM: %s\n", __func__, hb_mc_strerror(err));
		hb_mc_manycore_exit(mc);
		return err;
	}
	trans->done = num_int;

	return err;
}

void read_re_mem (hb_mc_manycore_t *mc, uint32_t base_addr, int len) {
	uint32_t read_data;
	int err;
	for (size_t i = 0; i < len; i++) {
		hb_mc_npa_t npa = { .x = RE_DRAM_X, .y = RE_DRAM_Y, .epa = base_addr*4 + (i*4) };
		err = hb_mc_manycore_read_mem(mc, &npa,
					      &read_data, sizeof(read_data));
		if (err != HB_MC_SUCCESS) {
			bsg_pr_err("%s: failed to read A[%d] "
				   "from DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
				   i, RE_DRAM_X, RE_DRAM_Y, base_addr + i);
		}
		printf("Read result(%d) %x \n", i, read_data);
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

void conv_fp_wrt_wgt (hb_mc_manycore_t *mc, NN_layer nn, float *weight, float *bias, uint32_t base_addr) {
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
                write_dram_float(mc, addr, number);
                addr++;
            }
            // Weights
	        for (int z=0; z<4; z++) {
	            for (int pe=0; pe<16; pe++) {
	                for (int y=3; y>-1; y--) {
	                    for (int x=0; x<4; x++) {
                            index = x + y*8 + z*8*8+ (pe+pe_offset)*8*8*4;
                            number = weight[index];
                            write_dram_float(mc, addr, number);
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
                            write_dram_float(mc, addr, number);
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
                            write_dram_float(mc, addr, number);
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
                            write_dram_float(mc, addr, number);
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
                write_dram_float(mc, addr, number);
                addr++;
            }
            // Weights
	        for (int z=0; z<32; z++) {
	            for (int pe=0; pe<16; pe++) {
	                for (int y=3; y>-1; y--) {
	                    for (int x=0; x<4; x++) {
                            index = x + y*4 + z*4*4+ (pe+pe_offset)*4*4*32;
                            number = weight[index];
                            write_dram_float(mc, addr, number);
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
                write_dram_float(mc, addr, number);
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
                                write_dram_float(mc, addr, number);
                                addr++;
                            }
                        }
                    }
                }
            }
        }
    }
}


void fc_fp_wrt_wgt (hb_mc_manycore_t *mc, NN_layer fc, float *weight, float *bias, uint32_t base_addr) {
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

void conv_bp_wrt_wgt (hb_mc_manycore_t *mc, NN_layer nn, float *weight) {
	float number;
	int index;
    uint32_t addr = nn.wT_base_addr;
    
    if(nn.layer==0){
        bsg_pr_test_info("conv1_bp_wrt_act begin\n");
        // Action re-arrangement for conv1_dw
        // (1,84,84,4)->(64,4,23,18)
        float newA[64*4*23*18] = {0};
        bsg_pr_test_info("newA\n");
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
        bsg_pr_test_info("conv1_bp_wrt_act\n");

	    for (int ff=0; ff<4; ff++) {
	        for (int z=0; z<23; z++) {
	            for (int x=0; x<1; x++) {
	                for (int y=0; y<64; y++) {
	                    for (int i_base=2; i_base>-1; y--) {
	                        for (int i=i_base*6; i<i_base*6+6; i++) {
                                index = y + (x+ff)*64 + z*64*4 + i*64*4*23;
                                number = newA[index];
                                write_dram_float(mc, addr, number);
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
                    write_dram_float(mc, addr, number);
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
                                    write_dram_float(mc, addr, number);
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
                write_dram_float(mc, addr, number);
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
                                write_dram_float(mc, addr, number);
                                addr++;
                            }
                        }
                    }
                }
            }
        }
    }
}

void fc_bp_wrt_wgt (hb_mc_manycore_t *mc, NN_layer nn, float *w, float *wT) {
	float number;
	int index, index0, index1;
    uint32_t addr = nn.wT_base_addr;
	host_transpose(w, wT, nn.input_size, nn.output_size);
    
    if(nn.layer==4) {
        for (int z=0; z<32; z++) {
            for (int j=0; j<16; j++) {
                number = 0.0;
                write_dram_float(mc, addr, number);
                addr++;
            }
            for (int j=0; j<16; j++) {
                for (int i=0; i<18; i++) {
                    if (i>=4) {
                        number = 0.0;
                    }
                    else {
                        index = z+j*32 + i*(nn.input_size);
                        number = wT[index];
                    }
                    write_dram_float(mc, addr, number);
                    addr++;
                }
            }
        }
    }
    else {
        for (int repeat=0; repeat<7; repeat++) {
            // for (int slides=0; slides<29; slides++) {
            for (int slides=0; slides<1; slides++) {
                for (int z=0; z<32; z++) {
                    if (slides==0) {
                        for (int j=0; j<16; j++) {
                            number = 0.0;
                            write_dram_float(mc, addr, number);
                            addr++;
                        }
                    }
                    else {
                        if (z==0) {
                            number == 0.0;
                            write_dram_float(mc, addr, number);
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
                                index = index0*(nn.input_size) + index1;
                                number = wT[index];
                            }
                            write_dram_float(mc, addr, number);
                            addr++;
                        }
                    }
                }
            }
        }
    }
}

void wgt_transpose_and_write (hb_mc_manycore_t *mc, NN_layer fc, float *w, float *wT) {
	int in_act_size = fc.input_size;
	int out_act_size = fc.output_size;
	host_transpose(w, wT, in_act_size, out_act_size);
	fc.input_size = out_act_size;
	fc.output_size = in_act_size;
	float zero_bias[1000] = {0.0};
	fc_fp_wrt_wgt(mc, fc, wT, zero_bias, fc.wT_base_addr);
}

void read_dw (hb_mc_manycore_t *mc, float *dW, uint32_t base_addr, int row, int col){
	uint32_t addr=base_addr;
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
			/* bsg_pr_test_info("Read dW(%d) %.5f, host result %.5f \n", index, dW[index], dW_host[index]); */
			/* bsg_pr_test_info("dY_host %.5f, X_host %.5f \n", dY_host[i], X_host[j]); */
		}
		if (row<18) 
			addr += (18-row);
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

void nn_fp(hb_mc_manycore_t *mc, float *state, NN_layer *nn, int num_layer, float* results) {
    bsg_pr_test_info("========Write state to DRAM on device========\n");
	float number;
	uint32_t addr=STATE_ADDR;
	for (int z = 0; z < 4; z++) { 
	    for (int x = 0; x < 21; x++) { 
	        for (int y = 0; y < 84; y++) { 
	            for (int i = 0; i < 4; i++) { 
		            number = state[x*4+i + y*84 + z*84*84];
		            addr++; 
		            write_dram_float(mc, addr, number);
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
	read_dram(mc, FP_RST_ADDR+1, ACTION_SIZE, results, false);
}

void nn_bp(hb_mc_manycore_t *mc, float *dy, NN_layer *nn, int num_layer) {
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
		write_dram_float(mc, addr, number);
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
    bsg_pr_test_info("========55555====\n");
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

void dqn_act(hb_mc_manycore_t *mc, Transition *trans, NN_layer *nn, int num_layer, float epsilon) {
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

void dqn_train(hb_mc_manycore_t *mc, Transition *trans, NN_layer *nn, int num_layer, float gamma) {
	// FP
	// next state
	float next_values[ACTION_SIZE];
	int next_max_index = 0;
	nn_fp(mc, trans->next_state, nn, num_layer, next_values);

	for (int i = 0; i < ACTION_SIZE; i++) { 
		if (next_values[i] > next_values[next_max_index])
			next_max_index = i;
        bsg_pr_test_info("DRLP Train: next_value[%d]=%f\n", i, next_values[i]);
	}
	// state
	float state_values[ACTION_SIZE];
	nn_fp(mc, trans->state, nn, num_layer, state_values);
    for (int i = 0; i < ACTION_SIZE; i++) { 
        bsg_pr_test_info("DRLP Train: state_value[%d]=%f\n", i, state_values[i]);
    }

	// Loss function
	float target = trans->reward + gamma*next_values[next_max_index];
	float fc2_dy[ACTION_SIZE]={0.0};
	fc2_dy[next_max_index] = next_values[next_max_index] - state_values[next_max_index]; // MSE loss function
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
