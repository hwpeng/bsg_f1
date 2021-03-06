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

# This Makefile is for running cosimulation regression
#
# environment.mk verifies the build environment and sets the following
# makefile variables:
#
# TESTBENCH_PATH: The path to the testbench directory in the bsg_f1 repository
# LIBRAIRES_PATH: The path to the libraries directory in the bsg_f1 repository
# HARDARE_PATH: The path to the hardware directory in the bsg_f1 repository
# BASEJUMP_STL_DIR: Path to a clone of BaseJump STL
# BSG_MANYCORE_DIR: Path to a clone of BSG Manycore
# CL_DIR: Path to the directory of this AWS F1 Project
include ../environment.mk

# The following makefile fragment verifies that the tools and CAD environment is
# configured correctly.
#
# *environment.mk must be included before cadenv.mk*
include $(CL_DIR)/cadenv.mk

# simlibs.mk defines build rules for hardware and software simulation libraries
# that are necessary for running cosimulation. These are dependencies for
# regression since running $(MAKE) recursively does not prevent parallel builds
# of identical rules -- which causes errors.
#
include simlibs.mk

# -------------------- Arguments --------------------
#
# This Makefile has several optional "arguments" that are passed as Variables
#
# DEBUG: Opens the GUI during cosimulation. Default: 0
# TURBO: Disables VPD generation. Default: 1
# EXTRA_TURBO: Disables VPD Generation, and more optimization flags: Default 0
# 
# If you need additional speed, you can set EXTRA_TURBO=1 during compilation. 
# This is a COMPILATION ONLY option. Any subsequent runs, without compilation
# will retain this setting

DEBUG ?= 0
TURBO ?= 1 # Use if you don't want debug information
EXTRA_TURBO ?= 0 # Use at your own risk.

# targets.mk defines the targets (sub-directories) for regression in
# cosimulation
include $(REGRESSION_PATH)/targets.mk
.PHONY: clean %.clean regression $(TARGETS)

regression: $(TARGETS)
	@cat $(foreach tgt,$(TARGETS),$(tgt)/regression.log)
$(TARGETS): $(SIMLIBS)
	$(MAKE) -C $@ regression EXTRA_TURBO=$(EXTRA_TURBO)	\
		DEBUG=$(DEBUG) TURBO=$(TURBO)

clean: $(addsuffix .clean,$(TARGETS)) simlibs.clean
	rm -rf *.log *.jou
	rm -rf .cxl*
	rm -rf *.bak

%.clean:
	$(MAKE) -C $(basename $@) clean
	rm -rf .Xil

.DEFAULT_GOAL := help
help:
	@echo "Usage:"
	@echo "make {regression|clean|<subdirectory_name>}"
	@echo "      regression: Run all cosimulation regression tests"
	@echo "      <subdirectory_name>: Run all the regression tests for"
	@echo "             a specific sub-directory (Options are: $(TARGETS))"
	@echo "      clean: Remove all cosimulation build files"
