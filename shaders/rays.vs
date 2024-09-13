#version 330

in vec2 hbox_coord;
// in vec2 tex_coord;

in vec3 rays;  // cos theta, sin theta, length
in vec3 ray_colors;

layout (std140) uniform Matrices{
    mat4 camera;
};

uniform float size;
uniform vec2 position;


// out vec2 TexCoord;
out vec3 Color;

void main(){
    // rotate and then translate
    float pos_x = hbox_coord.x * rays.x * rays.z * size - hbox_coord.y * rays.y;
    float pos_y = hbox_coord.x * rays.y * rays.z * size + hbox_coord.y * rays.x;
    vec2 pos = position + vec2(pos_x, pos_y);
    // vec2 pos = 2*position/width/1000000 + vec2(pos_x, pos_y) *25 + size/1000000 + head_dir/10000;
    // vec2 pos = hbox_coord *size + width/100000;// + size + width/1000000;

    gl_Position = camera * vec4(pos, 0, 1.0);
    // TexCoord = tex_coord;
    Color = ray_colors;
}