#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D heatmapTexture;
uniform vec3[256] colormap;
// uniform float minVal;
// uniform float maxVal;
// uniform float scale;
uniform float negGamma;


vec3 heatmapGreen(float value) {
    // if (value <= 0.5)  // negative range
    //     value = value * -minVal + minVal;
    // else  // positive range
    //     value = value * maxVal;  // -> in minVal, maxVal range
    
    // value = (value) * (maxVal - minVal) + minVal;   // min val, max val range
    // float x = clamp(value / scale, -1.0, 1.0);  // -1, 1 range

    if (value < 0.5) value = pow(value*2, negGamma)/2;  // gamma on negative range

    float x = value * 255;  // 0, 255 range 
    int i = int(x);
    return colormap[i];
    // return vec3(255-value*10, 0, 0)/255 + 0.0001 * colormap[i];
}


void main()
{
    float value = texture(heatmapTexture, TexCoord).r;
    vec3 color = heatmapGreen(value);
    FragColor = vec4(color, 1.0);   // green only colormap
}
