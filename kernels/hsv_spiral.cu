__device__ float hue2rgb(float v1, float v2, float vH) {
    if (vH < 0) vH += 1;
    if (vH > 1) vH -= 1;
    if (vH <= 1.0 / 6.0) return v1 + (v2 - v1) * 6.0 * vH;
    if (vH <= 1.0 / 2.0) return v2;
    if (vH <= 2.0 / 3.0) return v1 + (v2 - v1) * ((2.0 / 3.0) - vH) * 6.0;
    return v1;
}

extern "C"
__global__ void hsv_spiral(const float* scalar_colors, float* visual_color, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float t = scalar_colors[idx];
    float h = t / 255.0;
    float l = (t > 0) ? (0.5 + 0.15 * cosf(t * 0.2)) : 0.0;

    float var_2 = (l < 0.5) ? (l * 2.0) : 1.0;
    float var_1 = 2.0 * l - var_2;

    float r = 255.0 * hue2rgb(var_1, var_2, h + 1.0 / 3.0);
    float g = 255.0 * hue2rgb(var_1, var_2, h);
    float b = 255.0 * hue2rgb(var_1, var_2, h - 1.0 / 3.0);

    // Write the RGB values to the output array
    visual_color[3 * idx]     = r;
    visual_color[3 * idx + 1] = g;
    visual_color[3 * idx + 2] = b;
}
