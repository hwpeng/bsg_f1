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

#pragma once

#include <stdio.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include "library_tests.h"

#define DRLP_X 3 
#define DRLP_Y 4
#define DRAM_X 3
#define DRAM_Y 5

#define DRLP_CFG_LEN  7
#define DRLP_CFG_ADDR {0x0000, 0x0004, 0x0008, 0x000C, 0x0010, 0x0014, 0x0018, 0x001C}
#define DRLP_RST_LEN  8
#define DRLP_RST_ADDR {0x0020, 0x0024, 0x0028, 0x002C, 0x0030, 0x0034, 0x0038, 0x003C}
#define DRLP_DONE_ADDR 0x0040

#define CONV1_ACT_ADDR 0
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

#define RMEM_ADDR0 0 
#define RMEM_ADDR1 12800
#define RMEM_ADDR2 17984 
#define RMEM_ADDR3 21120 
#define RMEM_ADDR4 21632 


#define RESULT_ADDR (3787*1024)
