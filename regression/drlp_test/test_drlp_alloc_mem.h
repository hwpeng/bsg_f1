void optim_malloc(hb_mc_device_t device, NN_layer *nn, int num_layers, eva_t *w_opt_eva, eva_t *dw_opt_eva, eva_t *w_new_opt_eva, eva_t *b_opt_eva, eva_t *db_opt_eva, eva_t *b_new_opt_eva) {
    for (int i = 0; i < num_layers; i++) {
        size_t w_size = nn[i].weight_size * sizeof(uint32_t);
        size_t b_size = nn[i].bias_size * sizeof(uint32_t);
        hb_mc_device_malloc(&device, w_size, &(w_opt_eva[i])); 
        hb_mc_device_malloc(&device, w_size, &(dw_opt_eva[i])); 
        hb_mc_device_malloc(&device, w_size, &(w_new_opt_eva[i])); 
        hb_mc_device_malloc(&device, b_size, &(b_opt_eva[i])); 
        hb_mc_device_malloc(&device, b_size, &(db_opt_eva[i])); 
        hb_mc_device_malloc(&device, b_size, &(b_new_opt_eva[i])); 
    }
}

void malloc_npa(hb_mc_device_t device, size_t size, hb_mc_npa_t *new_npa) {
    hb_mc_manycore_t *mc = device.mc;
    hb_mc_coordinate_t target = { .x = 1, .y = 1 };
    eva_t new_eva;
    hb_mc_device_malloc(&device, size, &new_eva); 
    hb_mc_eva_to_npa(mc, &default_map, &target, &new_eva, new_npa, &size);

    hb_mc_idx_t new_x, new_y;
    hb_mc_epa_t new_epa;

    new_x = hb_mc_npa_get_x(new_npa);
    new_y = hb_mc_npa_get_y(new_npa);
    new_epa = hb_mc_npa_get_epa(new_npa);
    
    bsg_pr_test_info("malloc: EVA 0x%x mapped to NPA (x: %d, y: %d, EPA, 0x%x)\n", hb_mc_eva_addr(&new_eva), new_x, new_y, new_epa);
}
