#version 410 core

layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 vertexNormal_modelspace;

out vec3 WorldPos;
out vec3 WorldNormal;

layout(std140) uniform MVP {
	mat4 V; // 0
	mat4 P; // 64
};

uniform mat4 M;

void main() {
	vec4 worldPos4 = M * vec4(vertexPosition_modelspace, 1.0);
	WorldPos = worldPos4.xyz;
	WorldNormal = mat3(transpose(inverse(M))) * vertexNormal_modelspace; // Transform normal
	gl_Position = P * V * worldPos4;
	// gl_Position = vec4(WorldPos * 10, 1.0);
}