extern "C" __global__
void trace_rays_grid(float* rays, float* positions, float* sizes, float* colors, 
                     int* cells, int* cell_counts, float* results,
                     int num_organisms, int num_cells
                    //  int* cells_hit, int* orgs_checked, int* cache_hits
                    ) {
    /*
    Positions is [N, 2], where the 2 coordinates are the location.
    Sizes is [N, 1] where the coordinate is the radius.
    Rays is [N, R, 3], where N = organisms, R = rays per organism, and the first two coordinates are the direction (unit vector),
    and the last coordinate is the max length of the ray.
    
    Cells is [S, S, M], where S is the # cells, and M is the maximum number of creatures in a given cell.
    cell_counts is [S, S], where the coordinate is the number of creatures in a given cell.

    Results is [N, R, 3], where the 3 coordinates are the color of the creature that the ray intersects with (0 if nothing
    */
    int organism = blockIdx.x;
    int ray = threadIdx.x;

    // N ~ 10_000, R ~ 32 
    // N*R * N intersections to check


    if (organism >= num_organisms || ray >= CFG_num_rays) {  // check that we are in bounds
        return;
    }
    int ray_idx = (organism * CFG_num_rays + ray) * 3;
    float ray_dir_x = rays[ray_idx + 0];  // extract ray information
    float ray_dir_y = rays[ray_idx + 1];
    float ray_len = rays[ray_idx + 2];

    int organism_idx = organism * 2;
    float our_loc_x = positions[organism_idx + 0];  // extract organism information
    float our_loc_y = positions[organism_idx + 1];

    float t_best = ray_len+1.0f;  // initialize the best t value to the max length of the ray

    #if CFG_use_cache
        // set up cache for checking (so we don't double check objects)
        // needs to be a power of 2, defines the number of low-order bits we check
        int check_array[CFG_cache_size];
        for (int i = 0; i < CFG_cache_size; i++) {
            check_array[i] = -1;
        }
        int mask = CFG_cache_size - 1;
    #endif

    // getting our starting and ending grid coordinates
    int x = int(our_loc_x / CFG_cell_size);
    int y = int(our_loc_y / CFG_cell_size);
    int end_x = int((our_loc_x + ray_len * ray_dir_x) / CFG_cell_size);
    int end_y = int((our_loc_y + ray_len * ray_dir_y) / CFG_cell_size);
    end_x = max(0, min(end_x, num_cells-1));
    end_y = max(0, min(end_y, num_cells-1));

    // A Fast Voxel Traversal Algorithm for Ray Tracing - John Amanatides, Andrew Woo
    // https://www.desmos.com/calculator/yh3hevkyc5
    float t_delta_x = CFG_cell_size / abs(ray_dir_x);
    float t_delta_y = CFG_cell_size / abs(ray_dir_y);
    float t_x = -(our_loc_x - x * CFG_cell_size) / ray_dir_x;
    float t_y = -(our_loc_y - y * CFG_cell_size) / ray_dir_y;
    float t_max_x = ray_dir_x > 0 ? t_delta_x + t_x : t_x;
    float t_max_y = ray_dir_y > 0 ? t_delta_y + t_y : t_y;
    int sx = x < end_x ? 1 : -1;
    int sy = y < end_y ? 1 : -1;
    bool found_intersect = false;

    while (!found_intersect) {
        // num objects in current cell of line
        if (x < 0 || x >= num_cells || y < 0 || y >= num_cells) break;  // if outside the grid
        
        int cell_idx = y * num_cells + x;
        int cell_count = min(cell_counts[cell_idx], CFG_max_per_cell-1);
        for (int i = 0; i < cell_count; i++) {  // for each object in the cell
            int other_organism = cells[(cell_idx) * CFG_max_per_cell + i];
            // if (organism == 25 && ray == 21) {
            //     cell_idx = cell_idx + 1 - 2*3 + 5;
            // }
            if (other_organism == organism) continue;  // don't check ourselves
            // orgs_checked[organism * CFG_num_rays + ray] += 1;

            #if CFG_use_cache
                // check the cache to see if we've already checked this one
                int cache_idx = other_organism & mask;
                if (check_array[cache_idx] == other_organism) {
                    // cache_hits[organism * CFG_num_rays + ray] += 1;
                    continue;
                }
                check_array[cache_idx] = other_organism; // we have checked someone
            #endif

            // get info about other organism
            float other_x = positions[other_organism * 2 + 0];
            float other_y = positions[other_organism * 2 + 1];
            float other_rad = sizes[other_organism];
            float sq_other_rad = other_rad * other_rad;

            // calculate distance from our center to their center
            float diff_x = other_x - our_loc_x;
            float diff_y = other_y - our_loc_y;
            float sq_center_dist = diff_x * diff_x + diff_y * diff_y;

            // if we are inside, we are done immediately
            if (sq_center_dist < sq_other_rad){
                for (int c = 0; c < 3; c++) {
                    // using ray_idx here is a bit fallacious, since technically the '3' in indexing (c.f. x,y,len)
                    // rays is different from the '3' in indexing results (c.f. colors)
                    results[ray_idx + c] = colors[other_organism * 3 + c];
                }
                return;
            }

            // dot product of direction from us to them and the ray direction
            float t_opt = diff_x * ray_dir_x + diff_y * ray_dir_y;
            if (t_opt < 0) continue;  // if ray points away, and we aren't inside, we don't hit

            float discriminant = t_opt * t_opt + sq_other_rad - sq_center_dist;
            if (discriminant < 0) continue;  // no intersection of ray since discriminant doesn't exist

            float t_intersect = t_opt - sqrt(discriminant);  // compute actual intersection t value
            if (t_intersect > ray_len) continue;  // too far away

            // if we get here, we have an intersection, because t_intesect is less than ray_len
            // if we find 1 intersection in a grid cell, then there is no point in checking subsequent
            // grid cells, so we can stop looping over cells. Just get the closest within this cell, and
            // then we are done.
            found_intersect = true;

            if (t_intersect < t_best) {  // closer than previous object
                for (int c = 0; c < 3; c++) {
                    // using ray_idx here is a bit fallacious, since technically the '3' in indexing (c.f. x,y,len)
                    // rays is different from the '3' in indexing results (c.f. colors)
                    results[ray_idx + c] = colors[other_organism * 3 + c];
                }
                t_best = t_intersect;
            }
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
        // cells_hit[organism * CFG_num_rays + ray] += 1;
    }
}
