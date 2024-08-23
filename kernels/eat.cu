extern "C" __global__ 
void eat(int* positions, float* eat_pcts, float* pct_eaten, float* sizes,  float* food_grid,  // read from
    float* food_grid_updates, float* alive_costs, float* energies, float* healths, float* ages,  // write to
    int num_creatures, int num_cells, float food_decr   // constants
    ) {
    /* 
    Read from:
        Positions: [N, 2], where N = num_creatures, where the coordinates are the cell position of the creature.

        eat_pcts: [N], where the coordinate is the percentage of food that the creature eats in a cell.

        pct_eaten: [S, S], where S = num_cells, where the coordinate is the percentage of food eaten in a cell.

        Sizes: [N], where the coordinate is the size of the creature.

        food_grid: [S, S], where the coordinate is the amount of food in a cell


    Write to:  
        food_grid_updates: [S, S], where the coordinate is the amount of food to add to a cell.
        (we need to do it separately because we can't write to the same memory we read from, and we later need to do growing on
        all cells at once, which would require a grid-wide syrchronization)

        alive_costs: [N], where the coordinate is the cost of being alive for the creature.
        (we store this so that we can do .sum() for computing the step size when we grow things)

        energies: [N], where the coordinate is the energy of the creature.

        healths: [N], where the coordinate is the health of the creature.

        ages: [N], where the coordinate is the age of the creature.

    Constants:
        num_creatures: number of creatures.
        num_cells: number of cells in the grid.
        food_decr: amount of food that a creature destroys just by being in a cell (doesn't get eaten) (we can't just use the CFG_eat_pct
                    because it changes mid simulation => preprocessor wouldn't work)
*/
    // Get the index of the current thread
    int creature = blockIdx.x * blockDim.x + threadIdx.x;
    if (creature >= num_creatures) {
        return;
    }

    int x = positions[creature * 2 + 0];
    int y = positions[creature * 2 + 1];
    float size = sizes[creature];
    float eat_pct = eat_pcts[creature];

    int cell_idx = y * num_cells + x;
    float pct_claimed = pct_eaten[cell_idx];

    float food = food_grid[cell_idx];
    if (pct_claimed > 1.) {
        food *= eat_pct / pct_claimed;   // if more creatures trying to eat than there is food, it gets split
    } else {
        food *= eat_pct;  // else, they get the full amount
    }
    if (food < 0.0){
        healths[creature] += food * CFG_neg_food_eat_mul;
        food = 0.0;
    }
    else{
        healths[creature] += food * CFG_food_health_recovery;
    }
    float alive_cost = CFG_alive_cost(size);
    float creature_food_decr = food_decr * eat_pct * CFG_food_cover_decr_pct;

    energies[creature] += food - alive_cost;
    alive_costs[creature] = alive_cost;
    atomicAdd(&food_grid_updates[cell_idx], food+food_decr);
    ages[creature] += 1;
}
