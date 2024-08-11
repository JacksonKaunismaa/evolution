#version 330

in vec2 hbox_coord;
in vec2 tex_coord;

in vec2 position;
in float activation;
// layout (std140) uniform Matrices{
//     mat4 camera;
// };
uniform float neuron_size;
uniform float aspect_ratio;

out vec2 TexCoord;
out float Activation;

void main(){
    // rotate and then translate
    vec2 reshaped_hbox = vec2(hbox_coord.x, hbox_coord.y * aspect_ratio);
    vec2 pos = position + reshaped_hbox * neuron_size;

    gl_Position = vec4(pos, 0.0, 1.0);
    TexCoord = tex_coord;
    Activation = activation;
}