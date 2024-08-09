#version 330

in vec2 hbox_coord;
in vec2 tex_coord;

in vec2 position;
in float size;
in vec2 head_dir;  // cos theta, sin theta
in vec3 color;

layout (std140) uniform Matrices{
    mat4 camera;
};


out vec2 TexCoord;
out vec3 Color;

void main(){
    // rotate and then translate
    float pos_x = hbox_coord.x * head_dir.x - hbox_coord.y * head_dir.y;
    float pos_y = hbox_coord.x * head_dir.y + hbox_coord.y * head_dir.x;
    vec2 pos = position + vec2(pos_x, pos_y) * size;
    // vec2 pos = 2*position/width/1000000 + vec2(pos_x, pos_y) *25 + size/1000000 + head_dir/10000;
    // vec2 pos = hbox_coord *size + width/100000;// + size + width/1000000;

    gl_Position = camera * vec4(pos, 0.0, 1.0);
    TexCoord = tex_coord;
    Color = color;
}