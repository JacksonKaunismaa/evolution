extern "C" __global__
void gridded_is_attacking(float* positions, float* sizes, float* colors, float* head_dirs,
                          int* cells, int* cell_counts, float* results,
                          int num_organisms, int num_cells) {
    /*
    Positions is [N, 2], where the 2 coordinates are the location.
    Sizes is [N, 1] where the coordinate is the radius.
    Colors is [N, 3], 3 coordinates for RGB values.
    Head_dirs is [N, 2], where the two coordinates are the direction of the head (unit vector).
    
    cells is [S, S, M], where S is the # cells, and M is the maximum number of creatures in a given cell.
    cell_counts is [S, S], where the coordinate is the number of creatures in a given cell.

    Results is [N, 2], where the 1st coordinate is an integer indicating how many things the
    organism is attacking, and the 2nd coordinate is a float indicating the amount of damage the organism
    is taking from other things attacking it.
    */

    int organism = blockIdx.x * blockDim.x + threadIdx.x;

    if (organism >= num_organisms) {  // Out of bounds
        return;
    }

    // extract organism information
    int organism_idx = organism * 2;
    float center_x = positions[organism_idx + 0];
    float center_y = positions[organism_idx + 1];
    float size = sizes[organism];
    // locally store the color of the organism
    float color[3] = {colors[organism * 3 + 0], colors[organism * 3 + 1], colors[organism * 3 + 2]};

    // get head coordinate of organism in real space
    float attacking_x = center_x + head_dirs[organism_idx + 0] * size;
    float attacking_y = center_y + head_dirs[organism_idx + 1] * size;
    // get coordinate of head in cell space
    int attacking_i = int(attacking_x / CFG_cell_size);
    int attacking_j = int(attacking_y / CFG_cell_size);

    // clamp to grid boundaries, if head is outside the grid (only centers are restricted to be inside it)
    attacking_i = max(0, min(num_cells - 1, attacking_i));
    attacking_j = max(0, min(num_cells - 1, attacking_j));

    int cell_idx = attacking_j * num_cells + attacking_i;
    // if attack bonus is big enough to hit everything in the cell, don't compute distance
    bool small_cells = CFG_cell_size*1.4f < CFG_attack_dist_bonus;

    // check all creatures that share the cell of the head
    for (int idx = 0; idx < cell_counts[cell_idx]; idx++) {
        int other = cells[cell_idx * CFG_max_per_cell + idx];
        if (other == organism || other >= num_organisms) {  // if out of bounds or self, skip
            continue;
        }

        float other_color[3] = {colors[other * 3 + 0], colors[other * 3 + 1], colors[other * 3 + 2]};
        float color_diff = 0.0f;
        for (int c = 0; c < 3; c++) {
            color_diff += fabs(color[c] - other_color[c]);
        }
        if (color_diff <= CFG_attack_ignore_color_dist) {  // dont attack colors that are close to your own
            continue;
        }
        int other_idx = other * 2;

        if (small_cells){
            // fallacious to use organism/other idx since the 2 in results is not the same 2 as in positions
            atomicAdd(&results[organism_idx + 0], 1);
            atomicAdd(&results[other_idx + 1], size * size * 1.5f);
            continue;
            
        }

        float other_center_x = positions[other_idx + 0];
        float other_center_y = positions[other_idx + 1];
        float sq_dist = (attacking_x - other_center_x) * (attacking_x - other_center_x) +
                        (attacking_y - other_center_y) * (attacking_y - other_center_y);
        float other_size = sizes[other];

        if (sq_dist < other_size * other_size + CFG_attack_dist_bonus) {  // +1 to give some leeway
            //fallacious to use organism/other idx here for same reason as above
            atomicAdd(&results[organism_idx + 0], 1);  // damage is a function of size
            atomicAdd(&results[other_idx + 1], CFG_attack_dmg(size));
        }
    }
}
