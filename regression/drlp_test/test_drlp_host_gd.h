#define hex(X) (*(int*)&X)
#define flt(X) (*(float*)&X)

int host_compare (float *expect, float *get, int size, char *name) {
    float max_ferror = 0; 
    float ferror = 0;
    int mismatch = 0; 
	for (int i = 0; i < size; i++) { 
		ferror = hb_mc_calculate_float_error (expect[i], get[i]); 
        max_ferror = fmax (max_ferror, ferror);        
        if (ferror > 0.003 && fabsf(expect[i]) > 0.00005) {
			bsg_pr_err(BSG_RED("Mismatch: ") "%s[%d]: %.16f\tExpected: %.16f\tRelative error: %.16f\n",
            name, i, get[i], expect[i], ferror);
        mismatch = 1;
		}
	}
    // bsg_pr_test_info ("BackProp MAX relative floating-point error: %e\n", max_ferror); 
	return mismatch;
}

void host_fc_fp (float *x, float *w, float *b, float *y, int relu, int x_num, int y_num) { 
	for (int i = 0; i < y_num; i++) {
		y[i] = b[i];
		for(int j = 0; j < x_num; j++) {
			y[i] += x[j]*w[j*y_num+i];
			// if (relu==1) {
			// 	float t0, t1, t2, t3;
			// 	t0 = x[j];
			// 	t1 = w[j*y_num+i];
			// 	t2 = y[i];
			// 	t3 = b[i];
			// 	printf("Host: y[%d](%x) += x[%d](%x)*w[%d](%x); bias=%x \n", i, hex(t2), j, hex(t0), j*y_num+i, hex(t1), hex(t3));
			// }
		}
		if ((relu==1) && (y[i]<0.0)) { 
			y[i]=0.0;
		}
	}
}

void host_fc_dx (float *dy, float *wT, float *x, float *dx, int relu, int x_num, int y_num) { 
	for (int i = 0; i < y_num; i++) {
		dx[i] = 0;
		for(int j = 0; j < x_num; j++) {
			dx[i] += dy[j]*wT[j*y_num+i];
			float t0, t1, t3;
			t0 = dx[i];
			t1 = dy[j];
			t3 = wT[j*y_num+i];  
            // printf("Host: dx[%d](%x) += dy[%d](%x)*wT[%d](%x) \n", i, hex(t0), j, hex(t1), j*y_num+i, hex(t3));
		}
		if ((relu==1) && (x[i]==0.0)) { 
			dx[i]=0.0;
		}
	}
}

void host_fc_dw (float *dy, float *x, float *dw, int x_num, int y_num) { 
	for (int i = 0; i < y_num; i++) {
		for(int j = 0; j < x_num; j++) {
			dw[i*x_num+j] = dy[i]*x[j];
			float t0, t1, t3;
			t0 = dy[i];
			t1 = x[j];
			t3 = dw[i*x_num+j];  
            // printf("Host: dw[%d](%x) = dy[%d](%x)*x[%d](%x) \n", i*x_num+j, hex(t3), i, hex(t0), j, hex(t1));
		}
	}
}

void host_transpose (float *w, float *wT, int row, int col) { 
	for (int i = 0; i < row; i++) {
		for(int j = 0; j < col; j++) {
			wT[j*row+i] = w[i*col+j];
		}
	}
}

void host_fp (float *state, float *fc1_w, float* fc1_b, float *fc2_w, float* fc2_b, float *fc1_y, float *results, int state_size, int fc1_y_size, int action_size) {
	host_fc_fp(state, fc1_w, fc1_b, fc1_y, 1, state_size, fc1_y_size);
	host_fc_fp(fc1_y, fc2_w, fc2_b, results, 0, fc1_y_size, action_size);
	// for (int i = 0; i < ACTION_SIZE; i++) {
	// 	printf("Host results[%d] = %f \n", i, host_results[i]);
	// }
}

void host_bp (float *fc2_dy, float *fc1_y, float *fc2_wT, float *state, float *fc2_dw, float *fc1_dw, int state_size, int fc1_y_size, int action_size) {
	float fc2_dx[fc1_y_size];
	host_fc_dw(fc2_dy, fc1_y, fc2_dw, fc1_y_size, action_size);
	host_fc_dx(fc2_dy, fc2_wT, fc1_y, fc2_dx, 1, action_size, fc1_y_size);
	host_fc_dw(fc2_dx, state, fc1_dw, state_size, fc1_y_size);
}

void host_optimizer (float *w_new, float *w, float *gd, float gamma, int N) { 
        for (int i = 0; i < N; i ++) { 
        	w_new[i] = w[i] - gamma*gd[i];
            // if (i==28)
                // printf("w_new[%d]: %.16f\tw: %.16f\tgamma:%.5f\tdw: %.16f\n", 
                    // i, w_new[i], w[i], gamma, gd[i]);
        }
}

int host_train (float *state, float *next_state, float reward, float done, uint32_t action, float *fc1_w, float *fc1_b, float *fc2_w, float *fc2_b, float *fc2_wT, float *fc2_dw, float *fc1_dw, int state_size, int fc1_y_size, int action_size) {
	float gamma=0.95;

	// FP
	float next_values[action_size];
	float fc1_y[fc1_y_size];
	float state_values[action_size];
	int next_max_index = 0;
	// next_state
	host_fp(next_state, fc1_w, fc1_b, fc2_w, fc2_b, fc1_y, next_values, state_size, fc1_y_size, action_size);

	for (int i = 0; i < action_size; i++) { 
		if (next_values[i] > next_values[next_max_index])
			next_max_index = i;
        // bsg_pr_test_info("Host Train: next_value[%d]=%f\n", i, next_values[i]);
	}
	// state
	host_fp(state, fc1_w, fc1_b, fc2_w, fc2_b, fc1_y, state_values, state_size, fc1_y_size, action_size);
    for (int i = 0; i < action_size; i++) { 
        // bsg_pr_test_info("Host Train: state_value[%d]=%f\n", i, state_values[i]);
    }

	// Loss function
	float target;
    if (done == 0.0)
        target = reward + gamma*next_values[next_max_index];
    else
        target = reward;
	float fc2_dy[2]={0.0};
	fc2_dy[action] = state_values[action] - target; // MSE loss function
    // bsg_pr_test_info("HOST Train: reward=%f\n", reward);
    for (int i = 0; i < action_size; i++) { 
        // bsg_pr_test_info("Host Train: fc2_dy[%d]=%f\n", i, fc2_dy[i]);
    }

	// BP
	float host_fc2_dw[fc1_y_size*action_size];
	float host_fc1_dw[state_size*fc1_y_size];
	host_bp(fc2_dy, fc1_y, fc2_wT, state, host_fc2_dw, host_fc1_dw, state_size, fc1_y_size, action_size);

	// compare
	int mismatch=0;
	mismatch = host_compare (host_fc2_dw, fc2_dw, (fc1_y_size*action_size), "fc2 dw");
	mismatch = host_compare (host_fc1_dw, fc1_dw, (state_size*fc1_y_size), "fc1 dw");
	return mismatch;
}



