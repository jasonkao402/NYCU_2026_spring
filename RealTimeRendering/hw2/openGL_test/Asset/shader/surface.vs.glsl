#version 410 core
// TODO

layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 vertexNormal_modelspace;

out vec3 WorldNormal;
out vec3 WorldPos;
out vec3 DebugNormal;
out vec3 N;
out vec3 L;
out float NdotL;

layout(std140) uniform MVP {
	mat4 V; // 0
	mat4 P; // 64
};

uniform mat4 M;

void main() {
	N = normalize(WorldNormal);
    L = normalize(LightPos - WorldPos);
	NdotL = dot(N, L);
	gl_Position = P * V * M * vec4(vertexPosition_modelspace, 1);
	WorldPos = vec3(M * vec4(vertexPosition_modelspace, 1.0));
	WorldNormal = mat3(transpose(inverse(M))) * vertexNormal_modelspace; // Transform normal
	DebugNormal = vertexNormal_modelspace; // For debugging normal transformation
}