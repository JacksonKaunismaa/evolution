#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
in vec3 Color;

uniform sampler2D spriteTexture;
// uniform sampler2D circleTexture;


void main()
{
    vec4 texture_color = texture(spriteTexture, TexCoord);
    // vec4 circle_color = texture(circleTexture, TexCoord);
    // texture_color.a = 1.0;
    vec4 color = vec4(Color / 255., 1.0);
    if (texture_color.a < 0.1)
        discard;
    // FragColor = (texture_color + circle_color) * color;//* circle_color; //+ vec4(0.0, 1.0, 0.0, 1.0);
    FragColor = (texture_color) * color;//* circle_color; //+ vec4(0.0, 1.0, 0.0, 1.0);

}
