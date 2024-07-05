extern "C" __global__
void ray_trace(float* rays, float* positions, float* sizes, float* colors, float* results,
               int num_organisms, int num_rays) {
    /* Rays are [N, R, 3], where N = organisms, R = rays per organism, and the
    the first two coordinates are the direction (unit vector), and the last coordinate
    is the max length of the ray.
    
    Positions are [N, 2] where the 2 coordinates are the location.
    Sizes is [N, 1] where the coordinate is the radius.
    Colors is 3 coordinates for RGB values.
    
    Results are [N, R, 3], where the coordinates correspond to RGB colors of the think the ray hits
      e.g. 0 = nothing, and 1-255 are creatures of a particular color (based on genome.*/
    int organism = blockIdx.x;
    int ray = threadIdx.x;

    // N ~ 10_000, R ~ 32 
    // N*R * N intersections to check


    if (organism >= num_organisms || ray >= num_rays) {  // check that we are in bounds
        return;
    }
    int ray_idx = (organism * num_rays + ray) * 3;
    float ray_dir_x = rays[ray_idx + 0];  // extract ray information
    float ray_dir_y = rays[ray_idx + 1];
    float ray_len = rays[ray_idx + 2];

    int organism_idx = organism * 2;
    float our_loc_x = positions[organism_idx + 0];  // extract organism information
    float our_loc_y = positions[organism_idx + 1];

    float t_best = ray_len+1.0f;

    for (int other = 0; other < num_organisms; other++) {   // check all possible other organisms
        if (other == organism) continue;

        int other_idx = other * 2;
        float other_loc_x = positions[other_idx + 0];   // extract info about other organism
        float other_loc_y = positions[other_idx + 1];
        float other_rad = sizes[other];

        // scalar multiple of ray that is closest
        float t_opt = (other_loc_x - our_loc_x) * ray_dir_x + (other_loc_y - our_loc_y) * ray_dir_y;
        t_opt = fminf(fmaxf(t_opt, 0.0f), ray_len); // clamp to ray length

        float sq_closest_dist = (our_loc_x + t_opt * ray_dir_x - other_loc_x) * (our_loc_x + t_opt * ray_dir_x - other_loc_x) +
                                (our_loc_y + t_opt * ray_dir_y - other_loc_y) * (our_loc_y + t_opt * ray_dir_y - other_loc_y);

        if (sq_closest_dist > other_rad * other_rad) continue; // if we don't intersect the circle, skip

        // using ray_idx here is a bit fallacious, since technically the '3' in indexing (c.f. x,y,len)
        // rays is different from the '3' in indexing results (c.f. colors)
        if (t_opt < t_best) {  // check if its the best so far
            for (int c = 0; c < 3; c++) {  // copy in the color
                results[ray_idx + c] = colors[other * 3 + c];
            }
            t_best = t_opt;
        }
    }
}
