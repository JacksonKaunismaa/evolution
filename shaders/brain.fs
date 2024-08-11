#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D heatmapTexture;
uniform vec3[256] colormap;
// uniform float minVal;
// uniform float maxVal;
uniform float scale;
uniform float negGamma;


vec3 hotColdMap(float value) {
    if (value < 0) return vec3(0., 0., -value);  // blue for negative
    return vec3(value, 0., 0.);   // red for positive
}


void main()
{
    float value = texture(heatmapTexture, TexCoord).r;
    
    if (value == 0.0)
        discard;
    
    vec3 color = hotColdMap(value);

        // color = vec3(1.0, 1.0, 1.0);
    FragColor = vec4(color, 1.0);
}
