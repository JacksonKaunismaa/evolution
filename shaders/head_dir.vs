#version 330

in vec2 hbox_coord;
in vec2 head_dir;  // cos theta, sin theta

uniform float head_len;
uniform float head_start;
uniform vec3 color;

layout (std140) uniform Matrices{
    mat4 camera;
};

uniform float size;
uniform vec2 position;

// out vec2 TexCoord;
out vec3 Color;

void main(){
    // rotate and then translate
    float pos_x = hbox_coord.x * head_dir.x * head_len*size - hbox_coord.y * head_dir.y;
    float pos_y = hbox_coord.x * head_dir.y * head_len*size + hbox_coord.y * head_dir.x;
    vec2 pos = position + head_dir*head_start*size + vec2(pos_x, pos_y);

    gl_Position = camera * vec4(pos, 0, 1.0);
    // TexCoord = tex_coord;
    Color = color;
}