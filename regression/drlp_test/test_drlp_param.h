#define hex(X) (*(int*)&X)
#define flt(X) (*(float*)&X)

// Test game
#define FRAME_SIZE (84*84)
#define FRAME_NUM 4 
#define STATE_SIZE FRAME_SIZE*FRAME_NUM 
#define ACTION_SIZE 4

// NN config
#define CONV1_B_SIZE 32
#define CONV1_W_SIZE (8*8*4*32)
#define CONV1_Y_SIZE (20*20*32)
#define CONV2_B_SIZE 64
#define CONV2_W_SIZE (4*4*32*64)
#define CONV2_Y_SIZE (9*9*64)
#define CONV3_B_SIZE 64
#define CONV3_W_SIZE (3*3*64*64)
#define CONV3_Y_SIZE (7*7*64)
#define FC1_Y_SIZE 512
// #define FC1_Y_SIZE 18
#define FC1_B_SIZE FC1_Y_SIZE
#define FC1_W_SIZE CONV3_Y_SIZE*FC1_Y_SIZE
#define FC2_Y_SIZE ACTION_SIZE
#define FC2_B_SIZE FC2_Y_SIZE
#define FC2_W_SIZE FC1_Y_SIZE*FC2_Y_SIZE
// In DRLP
#define DRLP_X 3 
#define DRLP_Y 4

#define DRLP_CFG_LEN  7
#define DRLP_CFG_ADDR {0xFC0000, 0xFC0004, 0xFC0008, 0xFC000C, 0xFC0010, 0xFC0014, 0xFC0018, 0xFC001C}
#define DRLP_DONE_ADDR 0xFC0020
#define DRLP_RMEM_PREFIX 0xFE0000
#define DRLP_DRAM_CFG_ADDR 0xF80000 
#define RMEM_ADDR0 0 
#define RMEM_ADDR1 12800
#define RMEM_ADDR2 17984 
#define RMEM_ADDR3 21120 
#define RMEM_ADDR4 21632 
// #define RMEM_ADDR1 0
// #define RMEM_ADDR2 RMEM_ADDR1+CONV2_Y_SIZE 
// #define RMEM_ADDR3 RMEM_ADDR2+CONV3_Y_SIZE
// #define RMEM_ADDR4 RMEM_ADDR3+FC1_Y_SIZE 
// #define RMEM_ADDR0 RMEM_ADDR4+FC2_Y_SIZE

// DRAM
#define STATE_ADDR 1
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
#define FC2BP_DB_ADDR (5500*1024)
#define FC1BP_DB_ADDR (5501*1024)
#define CONV3BP_DB_ADDR (5502*1024)
#define CONV2BP_DB_ADDR (5503*1024)
#define CONV1BP_DB_ADDR (5504*1024)
#define FP_RST_ADDR (5505*1024)
#define DRLP_DRAM_SIZE FP_RST_ADDR+1024

// RL
#define EPISODE_MAX 2

#define LR 0.001
#define RE_MEM_SIZE 5000
#define RE_MEM_INIT_SIZE 1
#define TRANSITION_SIZE 2*STATE_SIZE+3
#define STEP_MAX 20
#define TRAIN_FREQ 1
#define MAX_EPSILON 0.01
#define MIN_EPSILON 0.01
#define EPSILON_DECAY 0.999

// Manycore
#define DRLP_X 3 
#define DRLP_Y 4

// Others
#define ONCHIP 0
#define DMA 1
#define HOST_COMPARE true


