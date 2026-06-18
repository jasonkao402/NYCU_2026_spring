#version 410 core

layout(location = 0) in vec3 vertexPosition_modelspace;

layout (std140) uniform MVP {
	mat4 V; // 0
	mat4 P; // 64
};

uniform mat4 M;

void main(){
	gl_Position =  P * V * M * vec4(vertexPosition_modelspace,1);
}