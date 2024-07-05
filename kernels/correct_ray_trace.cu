extern "C" __global__
void correct_ray_trace(float* rays, float* positions, float* sizes, float* colors, float* results,
                       int num_organisms) {
    /*Rays are [N, R, 3], where N = organisms, R = rays per organism, and the
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

    float t_best = ray_len+1.0f;

    for (int other = 0; other < num_organisms; other++) {  // check all possible other organisms
        if (other == organism) continue;

        int other_idx = other * 2;
        float other_loc_x = positions[other_idx + 0];   // extract info about other
        float other_loc_y = positions[other_idx + 1];
        float other_rad = sizes[other];
        float sq_other_rad = other_rad * other_rad;

        float sq_center_dist = (other_loc_x - our_loc_x) * (other_loc_x - our_loc_x) +
                               (other_loc_y - our_loc_y) * (other_loc_y - our_loc_y);

        if (sq_center_dist < sq_other_rad) {  // if we are inside it, the ray definitely hits
            for (int c = 0; c < 3; c++) {
                // using ray_idx here is a bit fallacious, since technically the '3' in indexing (c.f. x,y,len)
                // rays is different from the '3' in indexing results (c.f. colors)
                results[ray_idx + c] = colors[other * 3 + c];
            }
            return;
        }
        
        // we have sqrt(sq_center_dist) - other_rad > ray_len  => too far
        // if (sqrtf(sq_center_dist) - other_rad > ray_len) continue;  // maybe enable this, maybe don't

        // dot product of direction from us to them and the ray direction
        float t_opt = (other_loc_x - our_loc_x) * ray_dir_x + (other_loc_y - our_loc_y) * ray_dir_y;

        if (t_opt < 0) continue;  // if ray points away, and we aren't inside, we don't hit

        float discriminant = t_opt * t_opt + sq_other_rad - sq_center_dist;
        if (discriminant < 0) continue;  // no intersection of ray since discriminant doesn't exist

        float t_intersect = t_opt - sqrtf(discriminant);  // t value of intersection
        if (t_intersect > ray_len) continue;  // too far away

        if (t_intersect < t_best) {
            for (int c = 0; c < 3; c++) {
                // using ray_idx here is a bit fallacious, since technically the '3' in indexing (c.f. x,y,len)
                // rays is different from the '3' in indexing results (c.f. colors)
                results[ray_idx + c] = colors[other * 3 + c];
            }
            t_best = t_intersect;
        }
    }
}
