#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
in float Activation;

uniform sampler2D spriteTexture;


vec3 hotColdMap(float value) {
    if (value < 0) return vec3(0., 0., -value);  // blue for negative
    return vec3(value, 0., 0.);   // red for positive
}

void main()
{
    vec4 texture_color = texture(spriteTexture, TexCoord);
    vec3 color = hotColdMap(Activation);
    vec4 color4 = vec4(color, 1.0);
    if (texture_color.a < 0.1)
        discard;
    FragColor = color4;
}
