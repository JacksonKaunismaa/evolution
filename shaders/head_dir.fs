#version 330 core
out vec4 FragColor;

in vec3 Color;

void main()
{

    vec4 color = vec4(Color / 255., 0.5);
    if (Color.x == 0.0)
        color.a = 0.1; 
    FragColor = color;
}
