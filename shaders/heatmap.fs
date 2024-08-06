#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D heatmapTexture;
uniform vec3[256] colormap;
// uniform float minVal;
// uniform float maxVal;
uniform float scale;
uniform float negGamma;


vec3 heatmapGreen(float value) {
    float x = clamp(value / scale, -1.0, 1.0);  // -1, 1 range

    if (x < 0) x = -(pow(abs(x), negGamma));  // gamma on negative range

    x = x * 127 + 128;  // 0, 255 range 
    int i = int(x);
    return colormap[i];
    // return vec3(255-value*10, 0, 0)/255 + 0.0001 * colormap[i];
}


void main()
{
    float value = texture(heatmapTexture, TexCoord).r;
    vec3 color = heatmapGreen(value);
    FragColor = vec4(color, 0.3);   // green only colormap
}
