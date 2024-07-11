#version 330

in vec2 pos;
in vec2 tex_coord;

out vec2 TexCoord;

void main(){
    gl_Position = vec4(pos, 0.0, 1.0);
    TexCoord = tex_coord;
}