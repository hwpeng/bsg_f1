# Library (Host CUDA Tests)

This directory runs cosimulation regression tests of Manycore functionality on
F1. Each test is a .c/.h, or a .cpp/.hpp file pair, located in the `regression`
directory of the design.

Each test also has a corresponding .riscv binary file in the bsg_manycore/software/spmd/bsg_cuda_lite_runtime/ directory.

To add a test, see the instructions in `cl_manycore/regression/library/`. Tests
added to Makefile.tests in the `cl_manycore/regression/library/` will automatically
be run in this directory. 

To run all tests in an appropriately configured environment, run:

```make cosim``` 

The Makefile in this directory expects that the user has set `CL_DIR`,
`BSG_IP_CORES_DIR`, `BSG_MANYCORE_DIR`, and sourced the script hdk_setup.sh
inside of aws-fgpa.


