typedef struct {
	float state[STATE_SIZE];
	float next_state[STATE_SIZE];
	float reward;
	uint32_t action;
	uint32_t done;
} Transition;

typedef struct {
	int input_size;
	int output_size;
	int weight_size;
	int bias_size;
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
	uint32_t db_base_addr;
	int pe_on;
	int ymove;
	int zmove;
	int xmove;
	int img_w_count;
} NN_layer;


