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

#include "test_drlp_fpbp.h"
#include <math.h>

#define TEST_NAME "test_drlp_grad_descent"
#define ALLOC_NAME "default_allocator"
#define hex(X) (*(int*)&X)
#define flt(X) (*(float*)&X)

void host_fp (float *x, float *w, float *y, int x_num, int y_num) { 
	for (int i = 0; i < y_num; i++) {
		y[i] = 0;
		for(int j = 0; j < x_num; j++) {
			y[i] += x[j]*w[j*y_num+i];
		}
	}
	return;
}

void host_gen(float *x, float *w_real, float *y_real, int x_num, int y_num) {
	for (int i = 0; i < x_num; i++) {
		x[i] = rand()/(float)(RAND_MAX);
	}
	host_fp(x, w_real, y_real, x_num, y_num);
	return;
}

void host_dY (float *y_real, float *y_hat, float *dy, int y_num) { 
	for (int i = 0; i < y_num; i++) {
		dy[i] = y_real[i]-y_hat[i];
	}
	return;
}

void host_bp (float *dy, float *x, float *dw, int x_num, int y_num) { 
	for (int i = 0; i < x_num; i++) {
		for(int j = 0; j < y_num; j++) {
			dw[i*y_num+j] = x[i]*dy[j];
		}
	}
	return;
}
			
void host_optimizer (float *w_new, float *w, float *gd, float gamma, int N) { 
        for (int i = 0; i < N; i ++) { 
        	w_new[i] = w[i] + gamma*gd[i];
			// printf("w_new[%d]: %.32f\tw: %.32f\tdw: %.32f\n", 
			// 	i, w_new[i], w[i], gd[i]);
        }
        return;
}

float host_eval (float *w_real, float *w, int x_num, int y_num) {
	float x_test[x_num], y_real[y_num], y_hat[y_num];
	float err = 0.0;
	int max_y_real, max_y_hat;
	int N = 10;
	for (int i = 0; i < N; i++) {
		host_gen(x_test, w_real, y_real, x_num, y_num);
		host_fp(x_test, w, y_hat, x_num, y_num);
		for (int j = 0; j < y_num; j++) {
			err += fabs(y_real[j]-y_hat[j])/fabs(y_real[j]);
			/* printf("y_real[%d]: %.8f, y_hat[%d]: %.8f\n",  */
					/* j, y_real[j], y_hat[j]); */
		}
		/* printf("iter%d: %d %d \n", i, max_y_real, max_y_hat); */
	}
	/* printf("wrong %d \n", wrong); */
	return err/(float)(N*y_num);
}


int test_drlp_grad_descent (int argc, char **argv) {

	int rc;
	unsigned char *program_data;
	size_t program_size;
	char *bin_path, *test_name;

	bsg_pr_test_info("Running DRLP-CUDA gradient descent test!\n");

	/* srand(time(NULL));  */
	srand(0.1); 
	
	bin_path = "/mnt/users/ssd1/homes/huwan/bsg/bsg_bladerunner/bsg_manycore/software/spmd/bsg_cuda_lite_runtime/drlp_cuda/main.riscv";

	char *W_get_path, *X_path, *Y_real_path;
    uint32_t X_NUM = 540; //540
    uint32_t Y_NUM = 4;
    uint32_t W_NUM = Y_NUM*X_NUM;

    uint32_t X_base_addr = 1;
    uint32_t W_base_addr = 1024*5;
    uint32_t Y_base_addr = 1024*20;
    uint32_t dY_base_addr = 1024*25;
    uint32_t dW_base_addr = 1024*30;

    float X_host[X_NUM]; 
    /* float W_real[X_NUM][Y_NUM], W_host[X_NUM][Y_NUM];  */
    float W_real[W_NUM], W_host[W_NUM], dW_host[W_NUM]; 
    float Y_real[Y_NUM], Y_hat_host[Y_NUM], dY_host[Y_NUM]; 
	uint32_t num_int;
	float gamma = 0.001;

	uint32_t slides=X_NUM/18, pe_on = Y_NUM;// slides=30, pe_on=4
	uint32_t W_addr = W_base_addr;
	float number;

	// Random initialization
	for (int i = 0; i < X_NUM; i++) {
		for (int j = 0; j < Y_NUM; j++) {
			W_real[i*Y_NUM+j] = rand()/(float)(RAND_MAX);
			/* printf("W_real[%d]: %.32f \n", i*Y_NUM+j, W_real[i*Y_NUM+j]); */
			/* W_host[i*Y_NUM+j] = rand()/(float)(RAND_MAX); */
			W_host[i*Y_NUM+j] = 0.0;
		}
	}

    /*****************************************************************************************************************
    * Initialize device 
    ******************************************************************************************************************/
    hb_mc_device_t device;
    rc = hb_mc_device_init(&device, TEST_NAME, 0);
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to initialize device.\n");
            return rc;
    }
	hb_mc_manycore_t *mc = device.mc;

    /*****************************************************************************************************************
    * Write data to DRAM 
    ******************************************************************************************************************/
	bsg_pr_test_info("========Write init weight to DRAM on device========\n");
	for (int i = 0; i < slides; i++) {
		if (i==0) { 
			// Write bias
			for (int j = 0; j < pe_on; j++) {
				num_int = 0;
				hb_mc_npa_t npa = { .x = dram_coord_x, .y = dram_coord_y, .epa = W_addr*4};
				rc = hb_mc_manycore_write_mem(mc, &npa, &num_int, sizeof(num_int));
				W_addr += 1;
			}
		}
		else {
			// insert zero
			num_int = 0;
			hb_mc_npa_t npa = { .x = dram_coord_x, .y = dram_coord_y, .epa = W_addr*4};
			rc = hb_mc_manycore_write_mem(mc, &npa, &num_int, sizeof(num_int));
			W_addr += 1;
		}
		// weights
		for (int j = 0; j < pe_on; j++) {
			for (int k = 0; k < 18; k++) {
				number = W_host[(18*i+k)*pe_on+j];
				num_int = hex(number);
				hb_mc_npa_t npa = { .x = dram_coord_x, .y = dram_coord_y, .epa = W_addr*4};
				rc = hb_mc_manycore_write_mem(mc, &npa, &num_int, sizeof(num_int));
				if (rc != HB_MC_SUCCESS) {
					bsg_pr_err("%s: failed to write W = 0x%08" PRIx32 " "
						   "to DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
						   __func__,  num_int, dram_coord_x, dram_coord_y,
						   W_addr);
					hb_mc_manycore_exit(mc);
					return rc;
				}
				W_addr += 1;
			}
		}
	}
	// Training
	float rate;
	for (int i = 0; i < 10; i++) {
		if (i%1==0) {
			rate = host_eval(W_real, W_host, X_NUM, Y_NUM);
			printf("step %d: %.3f \n", i, rate);
			/* printf("W_host  %.32f \n", W_host[0]); */
		}
		host_gen(X_host, W_real, Y_real, X_NUM, Y_NUM);
		// printf("itet%d:\n Y_real=[", i);
		// for (int j = 0; j < Y_NUM; j++)
		// 	printf("%.8f\t", Y_real[j]);
		// printf("]\n Y_hat=[");
		// for (int j = 0; j < Y_NUM; j++)
		// 	printf("%.8f\t", Y_hat[j]);
		// printf("]\n");

		bsg_pr_test_info("========Write X to DRAM on device========\n");
		for (int i = 0; i < X_NUM; i++) { 
			number = X_host[i];
			num_int = hex(number);
			hb_mc_npa_t npa = { .x = dram_coord_x, .y = dram_coord_y, .epa = X_base_addr*4 + (i*4) };
			rc = hb_mc_manycore_write_mem(mc, &npa, &num_int, sizeof(num_int));
			if (rc != HB_MC_SUCCESS) {
				bsg_pr_err("%s: failed to write X[%d] = 0x%08" PRIx32 " "
					   "to DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
					   __func__, i, num_int,
					   dram_coord_x, dram_coord_y,
					   X_base_addr + i);
				hb_mc_manycore_exit(mc);
				return rc;
			}
		}

		bsg_pr_test_info("========Call DRLP FP========\n");
		fc_simple_test(mc, X_base_addr, W_base_addr, Y_base_addr);
		host_fp(X_host, W_host, Y_hat_host, X_NUM, Y_NUM);
		
		bsg_pr_test_info("========Read Y_hat from DRAM to host========\n");
		uint32_t read_data;
		float Y_hat[Y_NUM];
		for (size_t i = 0; i < Y_NUM; i++) {
			hb_mc_npa_t npa = { .x = dram_coord_x, .y = dram_coord_y , .epa = (Y_base_addr+1+i)*4 };
			rc = hb_mc_manycore_read_mem(mc, &npa,
						      &read_data, sizeof(read_data));
			Y_hat[i] = flt(read_data);
			if (rc != HB_MC_SUCCESS) {
				bsg_pr_err("%s: failed to read Y_hat[%d] "
					   "from DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
					   i, dram_coord_x, dram_coord_y, Y_base_addr+1+i);
			}
			bsg_pr_test_info("Read result(%d) %.5f, host result %.5f \n", i, Y_hat[i], Y_hat_host[i]);
		}

		bsg_pr_test_info("========dY = Y_real - Y_hat========\n");
		float dY[Y_NUM];
		host_dY(Y_real, Y_hat_host, dY_host, Y_NUM);
		host_dY(Y_real, Y_hat, dY, Y_NUM);

		bsg_pr_test_info("========Write dY to DRAM========\n");
		W_addr = dY_base_addr;
		for (size_t i = 0; i < Y_NUM; i++) {
			num_int = hex(dY[i]);
			hb_mc_npa_t npa = { .x = dram_coord_x, .y = dram_coord_y, .epa = W_addr*4};
			rc = hb_mc_manycore_write_mem(mc, &npa, &num_int, sizeof(num_int));
			if (rc != HB_MC_SUCCESS) {
				bsg_pr_err("%s: failed to write dY = 0x%08" PRIx32 " "
					   "to DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
					   __func__,  num_int, dram_coord_x, dram_coord_y,
					   W_addr);
				hb_mc_manycore_exit(mc);
				return rc;
			}
			W_addr++;
		}

		host_bp(dY_host, X_host, dW_host, X_NUM, Y_NUM);

		bsg_pr_test_info("========Call DRLP BP========\n");
		dw_simple_test(mc, dY_base_addr, X_base_addr-1, dW_base_addr);

		bsg_pr_test_info("========Read dW from DRAM to host========\n");
		// X dim first (540), then Y dim(4)
		float dW[W_NUM];
		W_addr = dW_base_addr+1;
		int index;
		for (int i = 0; i < Y_NUM; i++) {
			for (int j = 0; j < X_NUM; j++) {
				hb_mc_npa_t npa = { .x = dram_coord_x, .y = dram_coord_y , .epa = W_addr*4 };
				rc = hb_mc_manycore_read_mem(mc, &npa,
						      &read_data, sizeof(read_data));
				index = j*Y_NUM+i;
				dW[index] = flt(read_data);
				W_addr++;
				/* bsg_pr_test_info("Read dW(%d) %.5f, host result %.5f \n", index, dW[index], dW_host[index]); */
				/* bsg_pr_test_info("dY_host %.5f, X_host %.5f \n", dY_host[i], X_host[j]); */
			}
		}
		bsg_pr_test_info("========Read dW done!========\n");

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
    	eva_t W_device, dW_device, W_NEW_device; 
    	rc = hb_mc_device_malloc(&device, W_NUM* sizeof(uint32_t), &W_device); 
    	if (rc != HB_MC_SUCCESS) { 
    	        bsg_pr_err("failed to allocate memory on device.\n");
    	        return rc;
    	}
    	rc = hb_mc_device_malloc(&device, W_NUM * sizeof(uint32_t), &dW_device); 
    	if (rc != HB_MC_SUCCESS) { 
    	        bsg_pr_err("failed to allocate memory on device.\n");
    	        return rc;
    	}
    	rc = hb_mc_device_malloc(&device, W_NUM * sizeof(uint32_t), &W_NEW_device); 
    	if (rc != HB_MC_SUCCESS) { 
    	        bsg_pr_err("failed to allocate memory on device.\n");
    	        return rc;
    	}

    	/* Copy W & dW from host onto device DRAM (eva) */
		bsg_pr_test_info("========Copy from host to device DRAM (eva)========\n");
    	void *dst = (void *) ((intptr_t) W_device);
    	void *src = (void *) &W_host[0];
    	rc = hb_mc_device_memcpy (&device, dst, src, W_NUM * sizeof(uint32_t), HB_MC_MEMCPY_TO_DEVICE); 
    	if (rc != HB_MC_SUCCESS) { 
    	        bsg_pr_err("failed to copy memory to device.\n");
    	        return rc;
    	}
    	dst = (void *) ((intptr_t) dW_device);
    	src = (void *) &dW_host[0];
    	rc = hb_mc_device_memcpy (&device, dst, src, W_NUM * sizeof(uint32_t), HB_MC_MEMCPY_TO_DEVICE); 
    	if (rc != HB_MC_SUCCESS) { 
    	        bsg_pr_err("failed to copy memory to device.\n");
    	        return rc;
    	}

		/* Initialize values in W_NEW_device to 0. */
    	rc = hb_mc_device_memset(&device, &W_NEW_device, 0, W_NUM * sizeof(uint32_t));
    	if (rc != HB_MC_SUCCESS) { 
    	        bsg_pr_err("failed to set memory on device.\n");
    	        return rc;
    	} 

    	/* Define block_size_x/y: amount of work for each tile group */
    	/* Define tg_dim_x/y: number of tiles in each tile group */
    	/* Calculate grid_dim_x/y: number of tile groups needed based on block_size_x/y */
    	uint32_t block_size_x = W_NUM;
    	hb_mc_dimension_t tg_dim = { .x = 2, .y = 2}; 
    	hb_mc_dimension_t grid_dim = { .x = 1, .y = 1}; 

    	/* Prepare list of input arguments for kernel. */
    	int cuda_argv[6] = {W_device, dW_device, W_NEW_device, gamma, W_NUM, block_size_x};

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
		float W_NEW_host[W_NUM];
    	src = (void *) ((intptr_t) W_NEW_device);
    	dst = (void *) &W_NEW_host[0];
    	rc = hb_mc_device_memcpy (&device, (void *) dst, src, W_NUM * sizeof(uint32_t), HB_MC_MEMCPY_TO_HOST);
    	if (rc != HB_MC_SUCCESS) { 
    	        bsg_pr_err("failed to copy memory from device.\n");
    	        return rc;
    	}


    	/* Calculate the expected result using host code and compare. */ 
    	float W_NEW_expected[W_NUM]; 
		host_optimizer(W_NEW_expected, W_host, dW_host, gamma, W_NUM);

    	float max_ferror = 0; 
    	float ferror = 0;
    	int mismatch = 0; 
    	for (int i = 0; i < W_NUM; i++) {
			ferror = hb_mc_calculate_float_error (W_NEW_expected[i], W_NEW_host[i]); 
    	    max_ferror = fmax ( max_ferror, ferror);        
    	    if ( ferror > MAX_FLOAT_ERROR_TOLERANCE ) { 
    	            bsg_pr_err(BSG_RED("Mismatch: ") "C[%d]: %.32f\tExpected: %.32f\tRelative error: %.32f\n",
    	                               i,
    	                               W_NEW_host[i],
    	                               W_NEW_expected[i],
    	                               ferror);
    	            mismatch = 1;
    	    }
			W_host[i] = W_NEW_host[i];
    	} 
    	bsg_pr_test_info ("MAX relative FP error: %e\n", max_ferror); 
    	if (mismatch) { 
    	        return HB_MC_FAIL;
    	}

		bsg_pr_test_info("========Update weight to DRAM on device========\n");
		W_addr = W_base_addr;
		for (int i = 0; i < slides; i++) {
			if (i==0) { 
				// Write bias
				for (int j = 0; j < pe_on; j++) {
					W_addr += 1;
				}
			}
			else {
				// insert zero
				W_addr += 1;
			}
			// weights
			for (int j = 0; j < pe_on; j++) {
				for (int k = 0; k < 18; k++) {
					number = W_host[(18*i+k)*pe_on+j];
					num_int = hex(number);
					hb_mc_npa_t npa = { .x = dram_coord_x, .y = dram_coord_y, .epa = W_addr*4};
					rc = hb_mc_manycore_write_mem(mc, &npa, &num_int, sizeof(num_int));
					if (rc != HB_MC_SUCCESS) {
						bsg_pr_err("%s: failed to write W = 0x%08" PRIx32 " "
							   "to DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
							   __func__,  num_int, dram_coord_x, dram_coord_y,
							   W_addr);
						hb_mc_manycore_exit(mc);
						return rc;
					}
					W_addr += 1;
				}
			}
		}
	}
	
    /* Freeze the tiles and memory manager cleanup. */
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
	int rc = test_drlp_grad_descent(argc, argv);
	*exit_code = rc;
	bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
	return;
}
#else
int main(int argc, char ** argv) {
	bsg_pr_test_info(TEST_NAME " Regression Test (F1)\n");
	int rc = test_drlp_grad_descent(argc, argv);
	bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
	return rc;
}
#endif
