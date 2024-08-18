extern "C" __global__
void setup_eat_grid(int* positions, int* cell_counts, int num_creatures, int num_cells) {
    /*
    Positions is [N, 2], where the 2 coordinates are the location (in cells)
     
     cell_counts is [S, S], where the coordinate is the number of creatures in a given cell.
    */
    int creature = blockIdx.x * blockDim.x + threadIdx.x;

    if (creature >= num_creatures) {
        return;
    }

    int x = positions[creature * 2 + 0];
    int y = positions[creature * 2 + 1];

    int grid_idx = y * num_cells + x;
    atomicAdd(&cell_counts[grid_idx], 1);
}
