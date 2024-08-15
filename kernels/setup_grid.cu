extern "C" __global__
void setup_grid(float* positions, float* sizes, int* cells, int* cell_counts, 
                int num_creatures, int num_cells) {
    /*
    Positions is [N, 2], where the 2 coordinates are the location.
     Sizes is [N, 1] where the coordinate is the radius.
     
     cells is [S, S, M], where S is the # cells, and M is the maximum number of creatures in a given cell.
     cell_counts is [S, S], where the coordinate is the number of creatures in a given cell.
    */
    int creature = blockIdx.x * blockDim.x + threadIdx.x;

    if (creature >= num_creatures) {
        return;
    }

    float x = positions[creature * 2 + 0];
    float y = positions[creature * 2 + 1];
    float radius = sizes[creature];

    int min_x = max(0, int((x - radius) / CFG_cell_size));
    int max_x = min(num_cells - 1, int((x + radius) / CFG_cell_size));
    int min_y = max(0, int((y - radius) / CFG_cell_size));
    int max_y = min(num_cells - 1, int((y + radius) / CFG_cell_size));

    for (int i = min_x; i <= max_x; i++) {   // loop over all cells in the grid that this creature is in
        for (int j = min_y; j <= max_y; j++) {
            // find index to add creature at, atomicAdd returns the old value
            int grid_idx = j * num_cells + i;
            // int count = cell_counts[grid_idx] + 1;
            int old_count = atomicAdd(&cell_counts[grid_idx], 1);
            if (old_count >= CFG_max_per_cell - 1) {  // if old == max-1, then curr = max
                continue;
            }
            cells[grid_idx * CFG_max_per_cell + old_count] = creature;  // add the creature to the cell
        }
    }
}
