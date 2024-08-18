extern "C" __global__ 
void eat(int* positions, int* cell_counts, float* sizes,  float* food_grid,  // things we read from
    float* food_grid_updates, float* alive_costs, float* energies, float* ages,  // things we write to
    int num_creatures, int num_cells, float food_decr, float max_population   // constants
    ) {
    /* 
    Read from:
        Positions: [N, 2], where N = num_creatures, where the coordinates are the cell position of the creature.

        cell_counts: [S, S], where S = num_cells, where the coordinate is the number of creatures in a given cell.

        Sizes: [N], where the coordinate is the size of the creature.

        food_grid: [S, S], where the coordinate is the amount of food in a cell


    Write to:  
        food_grid_updates: [S, S], where the coordinate is the amount of food to add to a cell.
        (we need to do it separately because we can't write to the same memory we read from, and we later need to do growing on
        all cells at once, which would require a grid-wide syrchronization)

        alive_costs: [N], where the coordinate is the cost of being alive for the creature.
        (we store this so that we can do .sum() for computing the step size when we grow things)

        energies: [N], where the coordinate is the energy of the creature.

        ages: [N], where the coordinate is the age of the creature.

    Constants:
        num_creatures: number of creatures.
        num_cells: number of cells in the grid.
        food_decr: amount of food that a creature destroys just by being in a cell (doesn't get eaten) (we can't just use the CFG_eat_pct
                    because it changes mid simulation => preprocessor wouldn't work)
        max_population: maximum number of creatures that can be in a cell and still get full food from eating. 
*/
    // Get the index of the current thread
    int creature = blockIdx.x * blockDim.x + threadIdx.x;
    if (creature >= num_creatures) {
        return;
    }

    int x = positions[creature * 2 + 0];
    int y = positions[creature * 2 + 1];
    float size = sizes[creature];

    int cell_idx = y * num_cells + x;
    int count = cell_counts[cell_idx];

    float food = food_grid[cell_idx];
    if (count > max_population) {
        food *= 1. / count;   // if there are too many creatures, they split it equally
    } else {
        food *= CFG_eat_pct;  // else, they get the full amount
    }
    float alive_cost = CFG_alive_cost(size);

    alive_costs[creature] = alive_cost;
    energies[creature] += food - alive_cost;
    atomicAdd(&food_grid_updates[cell_idx], food+food_decr);
    ages[creature] += 1;
}
