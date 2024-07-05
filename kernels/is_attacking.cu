extern "C" __global__
void is_attacking(float* positions, float* sizes, float* colors, float* head_dirs, float* results,
                  int num_organisms) {
    /*
    Positions is [N, 2], where the 2 coordinates are the location.
    Sizes is [N, 1] where the coordinate is the radius.
    Colors is 3 coordinates for RGB values.

    Head_dirs is [N, 2], where the two coordinates are the direction of the head (unit vector).
    
    Results is [N, 2], where the 1st coordinate is an integer indicating how many things the 
    organism is attacking, and the 2nd coordinate is a float indicating the amount of damage the organism
    is taking from other things attacking it.
    */
    int org_attacking = blockIdx.x * blockDim.x + threadIdx.x;
    int org_vulnerable = blockIdx.y * blockDim.y + threadIdx.y;

    if (org_attacking >= num_organisms || org_vulnerable >= num_organisms){  // if out of bounds
        return;
    }

    if (org_attacking == org_vulnerable) {  // dont attack ourselves
        return;
    }

    int color_attack_index = org_attacking * 3;
    int color_vuln_index = org_vulnerable * 3;
    float color_diff = 0.0f;

    for (int c = 0; c < 3; c++) {  // if their color is too similar to ours, dont attack
        color_diff += fabs(colors[color_attack_index + c] - colors[color_vuln_index + c]);
    }
    if (color_diff <= CFG_attack_ignore_color_dist) {
        return;
    }

    float attack_size = sizes[org_attacking];
    float vuln_size = sizes[org_vulnerable];
    float sq_dist = 0.0f;
    int attack_index = org_attacking * 2;
    int vuln_index = org_vulnerable * 2;
    // compute squared distance from head to center of other organism
    for (int d = 0; d < 2; d++) {
        float dist = (positions[attack_index + d] + head_dirs[attack_index + d] * attack_size) - positions[vuln_index + d];
        sq_dist += dist * dist;
    }

    if (sq_dist < vuln_size * vuln_size + CFG_attack_dist_bonus) {  // +1 to give some leeway
    // need atomic since multiple threads compute attacks for the same organism
        atomicAdd(&results[org_attacking * 2 + 0], 1);  // attack damage is a function of size
        atomicAdd(&results[org_vulnerable * 2 + 1], CFG_attack_dmg(attack_size));   
    }
}
