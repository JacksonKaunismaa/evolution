import numpy as np
from numba import cuda
from numba import jit


@cuda.jit('void(float32[:, :, :], float32[:, :], float32[:, :], float32[:, :], float32[:, :, :])', 
          fastmath=True)
def ray_trace(rays, positions, sizes, colors, results):
    """Rays are [N, R, 3], where N = organisms, R = rays per organism, and the
    the first two coordinates are the direction (unit vector), and the last coordinate
    is the max length of the ray.
    
    Positions are [N, 2] where the 2 coordinates are the location.
    Sizes is [N, 1] where the coordinate is the squared radius.
    Colors is 3 coordinates for RGB values.
    
    Results are [N, R, 3], where the coordinates correspond to RGB colors of the think the ray hits
      e.g. 0 = nothing, and 1-255 are creatures of a particular color (based on genome."""

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
    our_loc = positions[organism]

    best_t = 0.

    for other in range(positions.shape[0]):  # check all the other possible creatures
        if other == organism:
            continue
        other_loc = positions[other]
        other_rad = sizes[other]

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
            results[organism, ray] = colors[other]
            best_t = opt_t


@cuda.jit('void(float32[:, :], float32[:, :],  float32[:, :], float32[:, :], float32[:, :])', 
          fastmath=True)
def is_attacking(positions, sizes, colors, head_dirs, results):
    """Positions is [N, 2], where the 2 coordinates are the location.
     Sizes is [N, 1] where the coordinate is the squared radius.
    Colors is 3 coordinates for RGB values.

    Head_dirs is [N, 2], where the two coordinates are the direction of the head (unit vector).
    
    Results is [N, 2], where the 1st coordinate is an integer indicating how many things the 
    organism is attacking, and the 2nd coordinate is a float indicating the amount of damage the organism
    is taking from other things attacking it."""
    
    org_attacking, org_vulnerable = cuda.grid(2)

    # check that we are in bounds
    if org_attacking >= positions.shape[0] or org_vulnerable >= positions.shape[0]:
        return
    
    if org_attacking == org_vulnerable:
        return
    
    org_attacking_color = colors[org_attacking]
    org_vulnerable_color = colors[org_vulnerable]

    # species that are close in color to one another can't attack each other
    color_diff = 0.
    for c in range(3):
        color_diff += abs(org_attacking_color[c] - org_vulnerable_color[c])
    if color_diff <= 3:
        return
    
    attacking_loc = positions[org_attacking] + \
                        head_dirs[org_attacking] * np.sqrt(sizes[org_attacking])
    vulnerable_loc = positions[org_vulnerable]

    dist = 0.0
    for d in range(2):
        dist += (attacking_loc[d] - vulnerable_loc[d]) ** 2

    if dist < sizes[org_vulnerable] + 1:   # +1 to give some leeway
        results[org_attacking, 0] += 1
        results[org_vulnerable, 1] += sizes[org_attacking]  # damage = size of attacker


    
    
    