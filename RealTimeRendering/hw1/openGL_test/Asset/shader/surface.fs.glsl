#version 410 core

in vec3 WorldPos;
out vec4 fragColor;

uniform vec3 colors; // object color
uniform vec3 lightPos;
uniform vec3 viewPos;

void main() {
	// TODO: implement Phong Shading
    
    fragColor = vec4(colors, 1.0);
}
