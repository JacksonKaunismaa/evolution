import numpy as np
from numba import cuda
from numba import jit


@cuda.jit('void(float32[:, :, :], float32[:, :], float32[:, :, :])')
def ray_trace(rays, objects, results):
    """Rays are [N, R, 3], where N = organisms, R = rays per organism, and the
    the first two coordinates are the direction (unit vector), and the last coordinate
    is the max length of the ray.
    
    Objects are [N, 4], where the first 2 coordinates describe the location, 3rd coordinate
    is the squared radius, and 4th coordinate is the color.
    
    Results are [N, R, 1, where the 1st coordinate is an integer corresponding to the type of thing the 
    ray hits. e.g. 0 = nothing, and 1-255 are objects of a particular color (based on genome."""

    # N ~ 10_000, R ~ 32 

    organism = cuda.blockIdx.x
    ray = cuda.threadIdx.x
    # for organism in range(rays.shape[0]):
    #     for ray in range(rays.shape[1]):
            

    # N*R * N intersections to check

    if organism >= rays.shape[0] or ray >= rays.shape[1]:  # check that we are in bounds
        return
    
    ray_dir = rays[organism, ray, :2]
    ray_len = rays[organism, ray, 2]
    our_loc = objects[organism, :2]

    best_t = 0.

    for other in range(objects.shape[0]):  # check all the other possible objects
        if other == organism:
            continue
        other_loc = objects[other, :2]
        other_rad = objects[other, 2]

        # calculate optimal squared distance 
        opt_t = 0.0
        for d in range(2):  # need to loop over dimensions
            opt_t += (other_loc[d] - our_loc[d]) * ray_dir[d]  # scalar multiple of the ray that is closest
        
        opt_t = min(max(opt_t, 0), ray_len)  # clamp to the ray length 

        closest_dist = 0.0
        for d in range(2):
            closest_dist += (our_loc[d] + opt_t * ray_dir[d] - other_loc[d]) ** 2

        if closest_dist > other_rad:
            continue

        # if its 0 then its the first object. Else, it needs to be closer than the previous object
        if results[organism, ray, 0] == 0 or opt_t < best_t:
            results[organism, ray, 0] = objects[other, 3]
            best_t = opt_t



    
    
    