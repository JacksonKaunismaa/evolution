import numpy as np
from numba import cuda
from numba import jit
import numba as nb
import math


@cuda.jit('void(float32[:, :, :], float32[:, :], float32[:], float32[:, :], float32[:, :, :])', 
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
    ray_len = math.sqrt(rays[organism, ray, 2])
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
            for c in range(3):
                results[organism, ray, c] = colors[other, c]
            best_t = opt_t



@cuda.jit('void(float32[:, :, :], float32[:, :], float32[:], float32[:, :], float32[:, :, :])', 
          fastmath=True)
def correct_ray_trace(rays, positions, sizes, colors, results):
    """Rays are [N, R, 3], where N = organisms, R = rays per organism, and the
    the first two coordinates are the direction (unit vector), and the last coordinate
    is the max squared length of the ray.
    
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
    ray_len = math.sqrt(rays[organism, ray, 2])
    our_loc = positions[organism]

    best_t = 0.

    for other in range(positions.shape[0]):  # check all the other possible creatures
        if other == organism:
            continue
        other_loc = positions[other]
        other_rad = sizes[other]

        center_dist = 0.0
        for d in range(2):
            center_dist += (other_loc[d] - our_loc[d])**2

        if center_dist < other_rad:    # then we are inside, then the ray definitely hits it
            for c in range(3):
                results[organism, ray, c] = colors[other, c]
            return
    
        # we have sqrt(center_dist) - sqrt(other_rad) > sqrt(ray_len)  => too far
        # square both sides: center_dist - 2*sqrt(center_dist*other_rad) + other_rad > ray_len
        # use upper bound on sqrt: center_dist*other_rad + 0. 25 >= sqrt(center_dist*other_rad)
        if center_dist - 2*other_rad*center_dist + other_rad + 0.25 > rays[organism, ray, 2]:
            continue

        # calculate t value of the closest point on the ray to the center of the circle
        opt_t = 0.0
        for d in range(2):  # need to loop over dimensions
            opt_t += (other_loc[d] - our_loc[d]) * ray_dir[d]  # scalar multiple of the ray that is closest

        if opt_t < 0:  # if negative it can't possible intersect it (since we aren't inside)
            continue
        
        discriminant = opt_t**2 + other_rad - center_dist
        if discriminant < 0:  # no intersection
            continue
        
        t_intersect = opt_t - math.sqrt(discriminant)  # compute actual intersection
        if t_intersect > ray_len:  # too far away
            continue

        # if its 0 then its the first object. Else, it needs to be closer than the previous object
        if results[organism, ray, 0] == 0 or opt_t < best_t:
            for c in range(3):
                results[organism, ray, c] = colors[other, c]
            best_t = opt_t


@cuda.jit('void(float32[:, :], float32[:], int32[:, :, :], int32[:, :], float32)', 
          fastmath=True)
def setup_grid(positions, sizes, cells, cell_counts, cell_size):
    """Positions is [N, 2], where the 2 coordinates are the location.
     Sizes is [N, 1] where the coordinate is the squared radius.
     
     cells is [S, S, M], where S is the # cells, and M is the maximum number of creatures in a given cell.
     cell_counts is [S, S], where the coordinate is the number of creatures in a given cell.
     Cell_size is the size of the cell in the grid.
     """
    creature = cuda.grid(1)
    if creature >= positions.shape[0]:
        return
    
    x, y = positions[creature]
    radius = math.sqrt(sizes[creature])
    grid_size = cells.shape[0]
    
    min_x = max(0, int((x - radius) // cell_size))
    max_x = min(grid_size - 1, int((x + radius) // cell_size))
    min_y = max(0, int((y - radius) // cell_size))
    max_y = min(grid_size - 1, int((y + radius) // cell_size))
    
    for i in range(min_x, max_x + 1):
        for j in range(min_y, max_y + 1):
            old_count = cuda.atomic.add(cell_counts, (j, i), 1)
            # print(f"Thread {creature} adding to cell {j}, {i} with count {old_count}")
            cells[j, i, old_count] = creature  # no idea why this -1 is necessary, but for some reason atomic add isn't working o.w.


@cuda.jit('void(float32[:, :, :], float32[:, :], float32[:], float32[:, :], int32[:, :, :], int32[:, :], float32, float32[:, :, :])',
            fastmath=True)
def trace_rays_grid(rays, positions, sizes, colors, cells, cell_counts, cell_size, results):
    """Positions is [N, 2], where the 2 coordinates are the location.
     Sizes is [N, 1] where the coordinate is the squared radius.
     Rays is [N, R, 3], where N = organisms, R = rays per organism, and the first two coordinates are the direction (unit vector),
       and the last coordinate is the max squared length of the ray.
    
    Cells is [S, S, M], where S is the # cells, and M is the maximum number of creatures in a given cell.
    cell_counts is [S, S], where the coordinate is the number of creatures in a given cell.
    Cell_size is the size of the cell in the partioned world.

    Results is [N, R, 3], where the 3 coordinates are the color of the creature that the ray intersects with (0 if nothing)."""
    # one thread per ray
    organism = cuda.blockIdx.x
    ray = cuda.threadIdx.x

    if organism >= rays.shape[0] or ray >= rays.shape[1]:  # check that we are in bounds
        return
    
    # extract information about the ray
    ray_x, ray_y = rays[organism, ray, :2]
    ray_len = math.sqrt(rays[organism, ray, 2])
    our_x, our_y = positions[organism]

    # set up cache for checking (so we don't double check objects)
    cache_size = 32   # needs to be a power of 2, defines the number of low-order bits we check
    check_array = cuda.local.array(cache_size, nb.int64)
    for i in range(cache_size):
        check_array[i] = -1
    mask = cache_size - 1
    
    # getting our starting and ending grid coordinates
    x = int(our_x // cell_size)
    y = int(our_y // cell_size)
    end_x = int((our_x + ray_len * ray_x) // cell_size)
    end_y = int((our_y + ray_len * ray_y) // cell_size)

    # Bresenham's line algorithm initialization
    dx = abs(end_x - x)
    dy = abs(end_y - y)
    sx = 1 if x < end_x else -1
    sy = 1 if y < end_y else -1
    err = dx - dy
    min_t = 0.

    grid_size = cells.shape[0]
    
    while True:
        # if outside the grid
        if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
            break

        # how many objects in this cell?
        cell_count = cell_counts[y, x]
        
        for i in range(cell_count): # for each object that is in this cell, do the distance check
            other_organism = cells[y, x, i]
            if other_organism == organism:  # don't check ourselves
                continue

            # check if we have already checked this organism
            cache_idx = other_organism & mask  # could get some issues with "dueling" cache entries
            if check_array[cache_idx] == other_organism:
                continue

            # extract info about the other organism
            other_x, other_y = positions[other_organism]
            other_rad = sizes[other_organism]
            
            # calculate distance from our center to their center
            center_dist = 0.0
            center_dist = (other_x - our_x)**2 + (other_y - our_y)**2

            # if we are inside, then we can quit immedieately
            if center_dist < other_rad:
                for c in range(3):
                    results[organism, ray, c] = colors[other_organism, c]
                return

            # calculate t value of the closest point on the ray to the center of the circle
            opt_t = (other_x - our_x) * ray_x + (other_y - our_y) * ray_y

            if opt_t < 0:  # if negative it can't possible intersect it (since we aren't inside)
                continue   # since the ray points in the opposite direction of the line connecting the 2 centers

            discriminant = opt_t**2 + other_rad - center_dist
            if discriminant < 0:  # no intersection
                continue
            
            t_intersect = opt_t - math.sqrt(discriminant)  # compute actual intersection
            if t_intersect > ray_len:  # too far away
                continue

            # if its 0 then its the first object. Else, it needs to be closer than the previous object
            if results[organism, ray, 0] == 0 or opt_t < min_t:
                for c in range(3):  # copy in the color
                    results[organism, ray, c] = colors[other_organism, c]
                min_t = opt_t
            
        # do Bresenhem line steps
        if x == end_x and y == end_y:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy



@cuda.jit('void(float32[:, :], float32[:],  float32[:, :], float32[:, :], float32[:, :])', 
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
    
    actual_len = math.sqrt(sizes[org_attacking])
    vulnerable_loc = positions[org_vulnerable]
    dist = 0.0
    for d in range(2):
        dist += (positions[org_attacking, d]+head_dirs[org_attacking, d]*actual_len - \
                    vulnerable_loc[d]) ** 2

    if dist < sizes[org_vulnerable] + 1:   # +1 to give some leeway
        results[org_attacking, 0] += 1
        results[org_vulnerable, 1] += sizes[org_attacking]  # damage = size of attacker