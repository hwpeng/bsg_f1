# Copyright (c) 2019, University of Washington All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
# Redistributions of source code must retain the above copyright notice, this list
# of conditions and the following disclaimer.
# 
# Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
# 
# Neither the name of the copyright holder nor the names of its contributors may
# be used to endorse or promote products derived from this software without
# specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This Makefile Fragment defines all of the rules for building
# cosimulation binaries

ORANGE=\033[0;33m
RED=\033[0;31m
NC=\033[0m

# This file REQUIRES several variables to be set. They are typically
# set by the Makefile that includes this makefile..
# 
# REGRESSION_TESTS: Names of all available regression tests.
ifndef REGRESSION_TESTS
$(error $(shell echo -e "$(RED)BSG MAKE ERROR: REGRESSION_TESTS is not defined$(NC)"))
endif

# SRC_PATH: The path to the directory containing the .c or cpp test files
ifndef SRC_PATH
$(error $(shell echo -e "$(RED)BSG MAKE ERROR: SRC_PATH is not defined$(NC)"))
endif

# EXEC_PATH: The path to the directory where tests will be executed
ifndef EXEC_PATH
$(error $(shell echo -e "$(RED)BSG MAKE ERROR: EXEC_PATH is not defined$(NC)"))
endif

# CL_DIR: The path to the root of the BSG F1 Repository
ifndef CL_DIR
$(error $(shell echo -e "$(RED)BSG MAKE ERROR: CL_DIR is not defined$(NC)"))
endif

# HARDWARE_PATH: The path to the hardware folder in BSG F1
ifndef HARDWARE_PATH
$(error $(shell echo -e "$(RED)BSG MAKE ERROR: HARDWARE_PATH is not defined$(NC)"))
endif

# TESTBENCH_PATH: The path to the testbenches folder in BSG F1
ifndef TESTBENCH_PATH
$(error $(shell echo -e "$(RED)BSG MAKE ERROR: TESTBENCH_PATH is not defined$(NC)"))
endif

# REGRESSION_PATH: The path to the regression folder in BSG F1
ifndef REGRESSION_PATH
$(error $(shell echo -e "$(RED)BSG MAKE ERROR: REGRESSION_PATH is not defined$(NC)"))
endif

# The following makefile fragment verifies that the tools and CAD environment is
# configured correctly.
include $(CL_DIR)/cadenv.mk

# The following variables are set by $(CL_DIR)/hdk.mk
#
# HDK_SHELL_DESIGN_DIR: Path to the directory containing all the AWS "shell" IP
# AWS_FPGA_REPO_DIR: Path to the clone of the aws-fpga repo
# HDK_COMMON_DIR: Path to HDK 'common' directory w/ libraries for cosimluation.
# SDK_DIR: Path to the SDK directory in the aws-fpga repo
include $(CL_DIR)/hdk.mk

# simlibs.mk defines build rules for hardware and software simulation libraries
# that are necessary for running cosimulation. These are dependencies for
# regression since running $(MAKE) recursively does not prevent parallel builds
# of identical rules -- which causes errors.
include $(TESTBENCH_PATH)/simlibs.mk

# -------------------- VARIABLES --------------------
# We parallelize VCS compilation, but we leave a few cores on the table.
NPROCS = $(shell echo "(`nproc`/4 + 1)" | bc)

# Name of the cosimulation wrapper system verilog file.
WRAPPER_NAME = cosim_wrapper

# libfpga_mgmt will be compiled in $(TESTBENCH_PATH)
LDFLAGS    += -lbsg_manycore_runtime -lm
LDFLAGS    += -L$(TESTBENCH_PATH) -Wl,-rpath=$(TESTBENCH_PATH)

# libbsg_manycore_runtime will be compiled in $(LIBRARIES_PATH)
LDFLAGS    += -L$(LIBRARIES_PATH) -Wl,-rpath=$(LIBRARIES_PATH)
# The bsg_manycore_runtime headers are in $(LIBRARIES_PATH) (for cosimulation)
INCLUDES   += -I$(LIBRARIES_PATH) 

# CSOURCES/HEADERS should probably go in some regression file list.
CSOURCES   += 
CHEADERS   += $(REGRESSION_PATH)/cl_manycore_regression.h
CDEFINES   += -DCOSIM -DVCS
CXXSOURCES += 
CXXHEADERS += $(REGRESSION_PATH)/cl_manycore_regression.h
CXXDEFINES += -DCOSIM -DVCS
CXXFLAGS   += -lstdc++

VCS_CFLAGS     += $(foreach def,$(CFLAGS),-CFLAGS "$(def)")
VCS_CDEFINES   += $(foreach def,$(CDEFINES),-CFLAGS "$(def)")
VCS_INCLUDES   += $(foreach def,$(INCLUDES),-CFLAGS "$(def)")
VCS_CXXFLAGS   += $(foreach def,$(CXXFLAGS),-CFLAGS "$(def)")
VCS_CXXDEFINES += $(foreach def,$(CXXDEFINES),-CFLAGS "$(def)")
VCS_LDFLAGS    += $(foreach def,$(LDFLAGS),-LDFLAGS "$(def)")
VCS_VFLAGS     += -M +lint=TFIPC-L -ntb_opts tb_timescale=1ps/1ps -lca -v2005 \
                -timescale=1ps/1ps -sverilog -full64 -licqueue +rad

# VCS Generates an executable file by compiling the $(SRC_PATH)/%.c or
# $(SRC_PATH)/%.cpp file that corresponds to the target test in the
# $(SRC_PATH) directory. % and %_debug targets differ by the arguments
# used to compile VCS that enable waveform generation and manycore logginc

# The following enables waveform generation for %_debug binaries (but not %)
$(EXEC_PATH)/%_debug: VCS_VFLAGS += -debug_pp +memcbk
# The following enables waveform generation for %_debug binaries (but
# not %) NOTE: undef_vcs_macro is a HACK!!!  `ifdef VCS is only used
# is in tb.sv top-level in the aws-fpga repository. This macro guards
# against generating vpd files, which slow down simulation. However,
# the only way to enable/disable the $vcdplusmemon and $vcdpluson
# system calls at vcs compile time is to use -undef_vcs_macro
$(EXEC_PATH)/%: VCS_VFLAGS += -undef_vcs_macro
$(EXEC_PATH)/% $(EXEC_PATH)/%_debug: $(SRC_PATH)/%.c $(CSOURCES) $(CHEADERS) $(SIMLIBS)
	SYNOPSYS_SIM_SETUP=$(TESTBENCH_PATH)/synopsys_sim.setup \
	vcs tb glbl -j$(NPROCS) $(WRAPPER_NAME) $< -Mdirectory=$@.tmp \
		$(VCS_CFLAGS) $(VCS_CDEFINES) $(VCS_INCLUDES) $(VCS_LDFLAGS) \
		$(VCS_VFLAGS) -o $@ -l $@.vcs.log -undef_vcs_macro

$(EXEC_PATH)/% $(EXEC_PATH)/%_debug: $(SRC_PATH)/%.cpp $(CXXSOURCES) $(CXXHEADERS) $(SIMLIBS)
	SYNOPSYS_SIM_SETUP=$(TESTBENCH_PATH)/synopsys_sim.setup \
	vcs tb glbl -j$(NPROCS) $(WRAPPER_NAME) $< -Mdirectory=$@.tmp \
		$(VCS_CXXFLAGS) $(VCS_CXXDEFINES) $(VCS_INCLUDES) $(VCS_LDFLAGS) \
		$(VCS_VFLAGS) -o $@ -l $@.vcs.log

$(REGRESSION_TESTS): %: $(EXEC_PATH)/%
$(addsuffix _debug,$(REGRESSION_TESTS)): %: $(EXEC_PATH)/%

test_loader: %: $(EXEC_PATH)/%
test_loader_debug: %: $(EXEC_PATH)/%

# To include a test in cosimulation, the user defines a list of tests in
# REGRESSION_TESTS. The following two lines defines a rule named
# <test_name>.rule that is a dependency in <test_name>.log. These custom
# rules can be used to build RISC-V binaries for SPMD or CUDA tests.
USER_RULES=$(addsuffix .rule,$(REGRESSION_TESTS))
$(USER_RULES):

# Likewise - we define a custom rule for <test_name>.clean
USER_CLEAN_RULES=$(addsuffix .clean,$(REGRESSION_TESTS))
$(USER_CLEAN_RULES):

compilation.clean: 
	rm -rf $(EXEC_PATH)/DVEfiles
	rm -rf $(EXEC_PATH)/*.daidir $(EXEC_PATH)/*.tmp
	rm -rf $(EXEC_PATH)/64 $(EXEC_PATH)/.cxl*
	rm -rf $(EXEC_PATH)/*.vcs.log $(EXEC_PATH)/*.jou
	rm -rf $(EXEC_PATH)/*.key $(EXEC_PATH)/*.vpd
	rm -rf $(EXEC_PATH)/vc_hdrs.h
	rm -rf .vlogansetup* stack.info*
	rm -rf $(REGRESSION_TESTS) test_loader

.PHONY: help compilation.clean $(USER_RULES) $(USER_CLEAN_RULES)

