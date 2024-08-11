#version 330 core
out vec4 FragColor;

// in vec2 TexCoord;
in vec3 Color;

// uniform sampler2D spriteTexture;


void main()
{
    // vec4 texture_color = texture(spriteTexture, TexCoord);

    vec4 color = vec4(Color / 255., 0.3);
    if (Color.x == 0.0)
        color.a = 0.1;    // if (texture_color.a < 0.1)
    //     discard;
    // FragColor = texture_color +color/999999999; //* color; //+ vec4(0.0, 1.0, 0.0, 1.0);
    FragColor = color;
}
