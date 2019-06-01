#include "test_shared_mem_load_store.h"

/*!
 * Runs a tile group shared memory load/store kernel. Loads a M * N matrix into tile group shared memory and stores it back to another location.
 * Grid dimensions are determines by how much of a load we want for each tile group (block_size_y/x)
 * This tests uses the software/spmd/bsg_cuda_lite_runtime/shared_mem_load_store/ Manycore binary in the dev_cuda_v4 branch of the BSG Manycore bitbucket repository.  
*/


int kernel_shared_mem_load_store () {
	fprintf(stderr, "Running the CUDA Shared Memory Load Store Kernel.\n\n");

	srand(time); 


	/*****************************************************************************************************************
	* Define the dimension of tile pool.
	* Define path to binary.
	* Initialize device, load binary and unfreeze tiles.
	******************************************************************************************************************/
	device_t device;
	uint8_t mesh_dim_x = 4;
	uint8_t mesh_dim_y = 4;
	uint8_t mesh_origin_x = 0;
	uint8_t mesh_origin_y = 1;
	eva_id_t eva_id = 0;
	char* elf = BSG_STRINGIFY(BSG_MANYCORE_DIR) "/software/spmd/bsg_cuda_lite_runtime" "/shared_mem_load_store/main.riscv";

	hb_mc_device_init(&device, eva_id, elf, mesh_dim_x, mesh_dim_y, mesh_origin_x, mesh_origin_y);




	/*****************************************************************************************************************
	* Allocate memory on the device for A, B and C.
	******************************************************************************************************************/
	uint32_t M = 64;
	uint32_t N = 64;

	eva_t A_in_device, A_out_device; 
	hb_mc_device_malloc(&device, M * N * sizeof(uint32_t), &A_in_device); /* allocate A_in[M][N] on the device */
	hb_mc_device_malloc(&device, N * N * sizeof(uint32_t), &A_out_device); /* allocate A_out[M][N] on the device */



	/*****************************************************************************************************************
	* Allocate memory on the host for A_in and initialize with random values.
	******************************************************************************************************************/
	uint32_t A_in_host[M * N]; /* allocate A[M][N] on the host */ 
	for (int i = 0; i < M * N; i++) { /* fill A with arbitrary data */
		A_in_host[i] = i; // rand() & 0xFFFF;
	}



	/*****************************************************************************************************************
	* Copy A_in from host onto device DRAM.
	******************************************************************************************************************/
	void *dst = (void *) ((intptr_t) A_in_device);
	void *src = (void *) &A_in_host[0];
	hb_mc_device_memcpy (&device, dst, src, (M * N) * sizeof(uint32_t), hb_mc_memcpy_to_device); /* Copy A_in to the device  */	


	/*****************************************************************************************************************
	* Define block_size_x/y: amount of work for each tile group
	* Define tg_dim_x/y: number of tiles in each tile group
	* Calculate grid_dim_x/y: number of tile groups needed based on block_size_x/y
	******************************************************************************************************************/
	uint32_t block_size_y = 32;
	uint32_t block_size_x = 16;

	uint8_t tg_dim_y = 2;
	uint8_t tg_dim_x = 2;

	uint32_t grid_dim_y = M / block_size_y;
	uint32_t grid_dim_x = N / block_size_x;


	/*****************************************************************************************************************
	* Prepare list of input arguments for kernel.
	******************************************************************************************************************/
	int argv[6] = {A_in_device, A_out_device, M, N, block_size_y, block_size_x};

	/*****************************************************************************************************************
	* Enquque grid of tile groups, pass in grid and tile group dimensions, kernel name, number and list of input arguments
	******************************************************************************************************************/
	hb_mc_grid_init (&device, grid_dim_x, grid_dim_y, tg_dim_x, tg_dim_y, "kernel_shared_mem_load_store", 6, argv);

	/*****************************************************************************************************************
	* Launch and execute all tile groups on device and wait for all to finish. 
	******************************************************************************************************************/
	hb_mc_device_tile_groups_execute(&device);
	


	/*****************************************************************************************************************
	* Copy result matrix back from device DRAM into host memory. 
	******************************************************************************************************************/
	uint32_t A_out_host[M * N];
	src = (void *) ((intptr_t) A_out_device);
	dst = (void *) &A_out_host[0];
	hb_mc_device_memcpy (&device, (void *) dst, src, (M * N) * sizeof(uint32_t), hb_mc_memcpy_to_host); /* copy A_out to the host */



	/*****************************************************************************************************************
	* Freeze the tiles and memory manager cleanup. 
	******************************************************************************************************************/
	hb_mc_device_finish(&device); 



		fprintf(stderr, "Expected Result:\n");
	for (int y = 0; y < M; y ++) { 
		for (int x = 0; x < N; x ++) { 
			fprintf(stderr, "%d\t", A_in_host[y * N + x]); 
		}
		fprintf(stderr, "\n");
	}

		
	fprintf(stderr, "Manycore Result:\n");
	for (int y = 0; y < M; y ++) { 
		for (int x = 0; x < N; x ++) { 
			fprintf(stderr, "%d\t", A_out_host[y * N + x]); 
		}
		fprintf(stderr, "\n");
	}

	
	// Compare matrices
	int mismatch = 0; 
	for (int y = 0; y < M; y ++) { 
		for (int x = 0; x < N; x ++) { 
			if ( A_in_host[y * N + x] != A_out_host[y * N + x]) { 
				mismatch = 1;
			}
		}
	}


	if (mismatch) { 
		fprintf(stderr, "Failed: matrix mismatch.\n");
		return HB_MC_FAIL;
	}
	fprintf(stderr, "Success: matrix match.\n");
	return HB_MC_SUCCESS;
}

#ifdef COSIM
void test_main(uint32_t *exit_code) {	
	bsg_pr_test_info("test_shared_mem_load_store Regression Test (COSIMULATION)\n");
	int rc = kernel_shared_mem_load_store();
	*exit_code = rc;
	bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
	return;
}
#else
int main() {
	bsg_pr_test_info("test_shared_mem_load_store Regression Test (F1)\n");
	int rc = kernel_shared_mem_load_store();
	bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
	return rc;
}
#endif
