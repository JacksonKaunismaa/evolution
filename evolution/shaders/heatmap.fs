#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D heatmapTexture;

// vec3 colormap(float value)
// {
//     // Apply a simple colormap (e.g., jet colormap)
//     float r = clamp(1.5 - abs(1.0 - 4.0 * (value - 0.5)), 0.0, 1.0);
//     float g = clamp(1.5 - abs(1.0 - 4.0 * (value - 0.25)), 0.0, 1.0);
//     float b = clamp(1.5 - abs(1.0 - 4.0 * (value)), 0.0, 1.0);
//     return vec3(r, g, b);
// }

void main()
{
    float value = texture(heatmapTexture, TexCoord).r;
    FragColor = vec4(0.0, value, 0.0, 1.0);   // green only colormap
}
