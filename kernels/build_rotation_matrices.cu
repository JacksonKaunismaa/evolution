extern "C" __global__ 
void build_rotation_matrices(float* outputs, float* sizes, float* energies, 
    int num_creatures, int num_outputs, float* rotation_matrices) {
    /* Outputs is [N, O], where N = organisms, O = num_outputs where 1st dimension is how much the 
    creature wants to move, 2nd dimension is how much the creature wants to rotate, and the rest
    is the memory of the creature.

    Sizes is [N], where N = organisms, where 1st dimension is the size of the creature.

    Energies is [N], where N = organisms, where 1st dimension is the energy of the creature.
     
    Rotation matrices is [N, 2, 2], where N = organisms, each index is a 2x2 rotation matrix.

    We write to energies and rotation_matrices (we defer the matrix multiply to pytorch).
*/
    // Get the index of the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) {
        return;
    }

    // Extract the rotation output from outputs
    float rotate_logit = outputs[idx * num_outputs + 1];
    float size = sizes[idx];
    float rotate = CFG_rotate_amt(rotate_logit, size);
    float rotate_energy = CFG_rotate_cost(rotate, size);

    // Deduct energy cost
    energies[idx] -= rotate_energy;

    // Compute cosine and sine of the rotation
    float sin_rotate, cos_rotate;
    sincosf(rotate, &sin_rotate, &cos_rotate);

    // write to rotation_matrices
    rotation_matrices[idx * 4 + 0] =   cos_rotate;
    rotation_matrices[idx * 4 + 1] =  -sin_rotate;
    rotation_matrices[idx * 4 + 2] =   sin_rotate;
    rotation_matrices[idx * 4 + 3] =   cos_rotate;
}
