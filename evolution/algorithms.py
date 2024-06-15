import numpy as np
from numba import cuda


@cuda.jit('void(float32[:, 4], float32[:, :], float32[:, 1])')
def ray_trace(rays, objects, results):
    """Rays are [N, R, 3], where N = organisms, R = rays per organism, and the
    the first two coordinates are the direction (unit vector), and the last coordinate
    is the max length of the ray.
    
    Objects are [N, 4], where the first 2 coordinates describe the location, 3rd coordinate
    is the squared radius, and 4th coordinate is the color.
    
    Results are [N, R, 2], where the 1st coordinate is an integer corresponding to the type of thing the 
    ray hits. e.g. 0 = nothing, and 1-255 are objects of a particular color (based on genome), and
    the second coordinate is the multiple of the ray that gets us closest to the object center."""

    # N ~ 10_000, R ~ 32 

    organism, ray = cuda.grid(2)

    # N*R * N intersections to check

    if organism >= rays.shape[0] or ray >= rays.shape[1]:  # check that we are in bounds
        return
    
    ray_dir = rays[organism, ray, :2]
    ray_len = rays[organism, ray, 2]
    our_loc = objects[organism, :2]

    for other in range(objects.shape[0]):  # check all the other possible objects
        other_loc = objects[other, :2]
        other_rad = objects[other, 2]

        # calculate optimal squared distance
        opt_t = np.dot(other_loc - our_loc, ray_dir)  # scalar multiple of the ray that is closest
        opt_t = min(max(opt_t, 0), ray_len)  # clamp to the ray length

        closest_diff = our_loc + opt_t * ray_dir - other_loc
        closest_dist = np.dot(closest_diff, closest_diff)

        if closest_dist > other_rad:
            continue

        # if its 0 then its the first object. Else, it needs to be closer than the previous object
        if results[organism, ray, 0] == 0 or opt_t < results[organism, ray, 1]:
            results[organism, ray, 0] = objects[other, 3]
            results[organism, ray, 1] = opt_t



    
    
    