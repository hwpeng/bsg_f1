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

#define TEST_NAME "test_drlp_cuda_simple"
#define ALLOC_NAME "default_allocator"
#define hex(X) (*(int*)&X)
#define flt(X) (*(float*)&X)

void host_optimizer (float *w, float *gd, float *w_new, float gamma, int N) { 
        for (int i = 0; i < N; i ++) { 
                w_new[i] = w[i] + gamma*gd[i];
        }
        return;
}

int test_drlp_cuda_simple (int argc, char **argv) {
	/* int err, r = HB_MC_FAIL; */
	int rc;
	unsigned char *program_data;
	size_t program_size;
	char *bin_path, *test_name;

	bsg_pr_test_info("Running DRLP CUDA simple test\n");

    srand(time(NULL)); 
	
	bin_path = "/mnt/users/ssd1/homes/huwan/bsg/bsg_bladerunner/bsg_manycore/software/spmd/bsg_cuda_lite_runtime/drlp_cuda/main.riscv";


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
    * Initialize device 
    ******************************************************************************************************************/
    uint32_t dY_NUM = 4;
    uint32_t X_NUM = 18*12*15; //(4,1)*(1,540)=(4,540)
    uint32_t dY_base_addr = 0;
    uint32_t X_base_addr = 1024;
    uint32_t R_base_addr = 1024*10;

    float dY_host[dY_NUM]; 
    float X_host[X_NUM]; 
	uint32_t num_int;

	bsg_pr_test_info("Generate data randomly on host and write to DRAM on device\n");
	for (int i = 0; i < dY_NUM; i++) {
		dY_host[i] = hb_mc_generate_float_rand();
		while (fabs(dY_host[i])>10e10 || fabs(dY_host[i])<10e-10) {
			dY_host[i] = hb_mc_generate_float_rand();
		}
		dY_host[i] = 1.0;
		num_int = hex(dY_host[i]);
		hb_mc_npa_t npa = { .x = dram_coord_x, .y = dram_coord_y, .epa = dY_base_addr*4 + (i*4) };
		bsg_pr_test_info("Write data %x to DRAM\n",num_int);
		rc = hb_mc_manycore_write_mem(mc, &npa, &num_int, sizeof(num_int));
		if (rc != HB_MC_SUCCESS) {
			bsg_pr_err("%s: failed to write dY[%d] = 0x%08" PRIx32 " "
				   "to DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
				   __func__, i, num_int,
				   dram_coord_x, dram_coord_y,
				   dY_base_addr + i);
			hb_mc_manycore_exit(mc);
			return rc;
		}
	}
	for (int i = 0; i < X_NUM; i++) { 
		X_host[i] = hb_mc_generate_float_rand();
		while (fabs(X_host[i])>10e10 || fabs(X_host[i])<10e-10) {
			X_host[i] = hb_mc_generate_float_rand();
		}
		X_host[i] = 0.5;
		num_int = hex(X_host[i]);
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

	bsg_pr_test_info("========Call DRLP!========\n");
	dw_simple_test(mc, dY_base_addr, X_base_addr, R_base_addr);

	
    /*****************************************************************************************************************
    * Allocate memory on the host for A & B and initialize with random values.
    ******************************************************************************************************************/
	float gamma = 0.1;
    uint32_t N = 16;
	uint32_t read_data;
    float W_host[N]; /* allocate W[N] on the host */ 
    float GD_host[N]; /* allocate GD[N] on the host */

	for (size_t i = 0; i < N; i++) {
		hb_mc_npa_t npa = { .x = dram_coord_x, .y = dram_coord_y , .epa = (R_base_addr+1)*4 + (i*4) };
		rc = hb_mc_manycore_read_mem(mc, &npa,
					      &read_data, sizeof(read_data));
		if (rc != HB_MC_SUCCESS) {
			bsg_pr_err("%s: failed to read A[%d] "
				   "from DRAM coord(%d,%d) @ 0x%08" PRIx32 "\n",
				   i, dram_coord_x, dram_coord_y, R_base_addr+1 + i);
		}
		bsg_pr_test_info("Read result(%d) %x \n", i, read_data);
        GD_host[i] = flt(read_data);
        /* W_host[i] = hb_mc_generate_float_rand(); */
		W_host[i] = 1.0;
	}


    rc = hb_mc_device_program_init(&device, bin_path, ALLOC_NAME, 0);
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to initialize program.\n");
            return rc;
    }
	else {
		bsg_pr_test_info("Initialize program %s. \n", bin_path);
	}

    /*****************************************************************************************************************
    * Allocate memory on the device for W, GD, and W_NEW
    ******************************************************************************************************************/

    eva_t W_device, GD_device, W_NEW_device; 
    rc = hb_mc_device_malloc(&device, N * sizeof(uint32_t), &W_device); /* allocate W[N] on the device */
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to allocate memory on device.\n");
            return rc;
    }


    rc = hb_mc_device_malloc(&device, N * sizeof(uint32_t), &GD_device); /* allocate GD[N] on the device */
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to allocate memory on device.\n");
            return rc;
    }


    rc = hb_mc_device_malloc(&device, N * sizeof(uint32_t), &W_NEW_device); /* allocate W_NEW[N] on the device */
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to allocate memory on device.\n");
            return rc;
    }


    /*****************************************************************************************************************
    * Copy W & GD from host onto device DRAM.
    ******************************************************************************************************************/
    void *dst = (void *) ((intptr_t) W_device);
    void *src = (void *) &W_host[0];
    rc = hb_mc_device_memcpy (&device, dst, src, N * sizeof(uint32_t), HB_MC_MEMCPY_TO_DEVICE); /* Copy W to the device  */ 
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to copy memory to device.\n");
            return rc;
    }


    dst = (void *) ((intptr_t) GD_device);
    src = (void *) &GD_host[0];
    rc = hb_mc_device_memcpy (&device, dst, src, N * sizeof(uint32_t), HB_MC_MEMCPY_TO_DEVICE); /* Copy GD to the device */ 
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to copy memory to device.\n");
            return rc;
    }

    /**********************************************************************/
    /* Initialize values in W_NEW_device to 0.                                */
    /**********************************************************************/
    rc = hb_mc_device_memset(&device, &W_NEW_device, 0, N * sizeof(uint32_t));
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to set memory on device.\n");
            return rc;
    } 

    /*****************************************************************************************************************
    * Define block_size_x/y: amount of work for each tile group
    * Define tg_dim_x/y: number of tiles in each tile group
    * Calculate grid_dim_x/y: number of tile groups needed based on block_size_x/y
    ******************************************************************************************************************/
    uint32_t block_size_x = N;

    hb_mc_dimension_t tg_dim = { .x = 2, .y = 2}; 

    hb_mc_dimension_t grid_dim = { .x = 1, .y = 1}; 


    /*****************************************************************************************************************
    * Prepare list of input arguments for kernel.
    ******************************************************************************************************************/
    int cuda_argv[6] = {W_device, GD_device, W_NEW_device, gamma, N, block_size_x};

    /*****************************************************************************************************************
    * Enquque grid of tile groups, pass in grid and tile group dimensions, kernel name, number and list of input arguments
    ******************************************************************************************************************/
    rc = hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_optimizer", 6, cuda_argv);
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to initialize grid.\n");
            return rc;
    }


    /*****************************************************************************************************************
    * Launch and execute all tile groups on device and wait for all to finish. 
    ******************************************************************************************************************/
    rc = hb_mc_device_tile_groups_execute(&device);
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to execute tile groups.\n");
            return rc;
    }

    /**********************************************************************/
    /* Copy result matrix back from device DRAM into host memory.         */
    /**********************************************************************/
    float W_NEW_host[N];
    src = (void *) ((intptr_t) W_NEW_device);
    dst = (void *) &W_NEW_host[0];
    rc = hb_mc_device_memcpy (&device, (void *) dst, src, N * sizeof(uint32_t), HB_MC_MEMCPY_TO_HOST);
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to copy memory from device.\n");
            return rc;
    }


    /**********************************************************************/
    /* Freeze the tiles and memory manager cleanup.                       */
    /**********************************************************************/
    rc = hb_mc_device_finish(&device); 
    if (rc != HB_MC_SUCCESS) { 
            bsg_pr_err("failed to de-initialize device.\n");
            return rc;
    }


    /**********************************************************************/
    /* Calculate the expected result using host code and compare.         */ 
    /**********************************************************************/
    float W_NEW_expected[N]; 
    host_optimizer (W_host, GD_host, W_NEW_expected, gamma, N); 

    float max_ferror = 0; 
    float ferror = 0;

    int mismatch = 0; 
    for (int i = 0; i < N; i++) {
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
    } 

    bsg_pr_test_info ("MAX relative FP error: %e\n", max_ferror); 

    if (mismatch) { 
            return HB_MC_FAIL;
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
	int rc = test_drlp_cuda_simple(argc, argv);
	*exit_code = rc;
	bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
	return;
}
#else
int main(int argc, char ** argv) {
	bsg_pr_test_info(TEST_NAME " Regression Test (F1)\n");
	int rc = test_drlp_cuda_simple(argc, argv);
	bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
	return rc;
}
#endif
