__device__ inline bool in_attack_range(float ray_origin_x, float ray_origin_y, float size, float ray_dir_x, float ray_dir_y,
                                float other_x, float other_y, float other_rad) {
    float sq_other_rad = other_rad * other_rad;
    float diff_x = other_x - ray_origin_x;
    float diff_y = other_y - ray_origin_y;
    float sq_center_dist = diff_x * diff_x + diff_y * diff_y;
    if (sq_center_dist < sq_other_rad) return true; // center is inside ray origin

    float t_opt = diff_x * ray_dir_x + diff_y * ray_dir_y;
    if (t_opt < 0) return false;  // ray points away from center

    float discriminant = t_opt * t_opt + sq_other_rad - sq_center_dist;
    if (discriminant < 0) return false;  // no intersection of ray and circle

    float t_intersect = t_opt - sqrt(discriminant); // compute intersection point t value
    // we know t_intersect > 0, but we start our ray origin from t = CFG_attack_range(0) * size, so we need to check
    // the upper bound only, and adjust the range to be the difference between the two bounds
    // t \in [CFG_attack_range(0)*size, CFG_attack_range(1)*size]
    return t_intersect <= (CFG_attack_range(1) - CFG_attack_range(0)) * size;
}


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
    float head_x = head_dirs[organism_idx + 0];
    float head_y = head_dirs[organism_idx + 1];

    // get coordinate of head in cell space
    float ray_origin_x = center_x + head_x * size * CFG_attack_range(0);
    float ray_origin_y = center_y + head_y * size * CFG_attack_range(0);
    int x = int(ray_origin_x / CFG_cell_size);
    int y = int(ray_origin_y / CFG_cell_size);
    int end_x = int((center_x + head_x * size * CFG_attack_range(1)) / CFG_cell_size);
    int end_y = int((center_y + head_y * size * CFG_attack_range(1)) / CFG_cell_size);
    end_x = max(0, min(num_cells - 1, end_x));
    end_y = max(0, min(num_cells - 1, end_y));
    x = max(0, min(num_cells - 1, x));
    y = max(0, min(num_cells - 1, y));

    float t_delta_x = CFG_cell_size / abs(head_x);
    float t_delta_y = CFG_cell_size / abs(head_y);
    float t_x = -(center_x - x * CFG_cell_size) / head_x;
    float t_y = -(center_y - y * CFG_cell_size) / head_y;
    float t_max_x = head_x > 0 ? t_delta_x + t_x : t_x;
    float t_max_y = head_y > 0 ? t_delta_y + t_y : t_y;
    int sx = x < end_x ? 1 : -1;
    int sy = y < end_y ? 1 : -1;
    bool found_intersect = false;

    // if attack bonus is big enough to hit everything in the cell, don't compute distance
    bool small_cells = CFG_cell_size*sqrt(2.0f) < CFG_attack_dist_bonus;

    // this implementation allows multi attacks if the attack ray crosses a cell boundary
    while (!found_intersect){
        if (x < 0 || x >= num_cells || y < 0 || y >= num_cells) break;  // if outside the grid
        
        int cell_idx = y * num_cells + x;
        int cell_count = min(cell_counts[cell_idx], CFG_max_per_cell-1);
        // check all creatures that share the cell of the head
        for (int idx = 0; idx < cell_count; idx++) {
            int other = cells[cell_idx * CFG_max_per_cell + idx];
            if (other == organism) continue;  // if self, skip

            // fallacious to use organism/other idx since the 2 in results is not the same 2 as in positions
            int other_idx = other * 2;

            if (!small_cells){  // only check range if small cells optimization doesn't apply
                if (!in_attack_range(ray_origin_x, ray_origin_y, size, head_x, head_y,
                                    positions[other_idx + 0], positions[other_idx + 1], sizes[other])) {
                    continue;
                }
            }
            
            // do_attack: {
            float other_color[3] = {colors[other * 3 + 0], colors[other * 3 + 1], colors[other * 3 + 2]};
            float color_diff = 0.0f;
            float cdif = 0.0f;
            for (int c = 0; c < 3; c++) {
                cdif = fabs(color[c] - other_color[c]);
                color_diff += min(cdif, 255.0f - cdif);
            }
            float color_diff_frac = color_diff / (255.0f*3.0f / 2.0f);
            //fallacious to use organism/other idx here for same reason as above
            atomicAdd(&results[organism_idx + 0], 1);  // damage is a function of size
            atomicAdd(&results[other_idx + 1], color_diff_frac * CFG_attack_dmg(size));
            // }
        }

        if (x == end_x && y == end_y) break;   // includes being outside the grid

        // do Voxel Traversal
        if (t_max_x < t_max_y) {
            t_max_x += t_delta_x;
            x += sx;
        } else {
            t_max_y += t_delta_y;
            y += sy;
        }
    }
}
