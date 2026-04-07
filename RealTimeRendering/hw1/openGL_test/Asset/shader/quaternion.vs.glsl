#version 410 core

layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 vertexNormal_modelspace;

out vec3 WorldNormal;
out vec3 WorldPos;
out vec3 DebugNormal;

layout(std140) uniform MVP {
	mat4 V; // 0
	mat4 P; // 64
};

uniform mat4 M;
uniform float currentTime = 0.0;
uniform float orbitRadius = 2.5;
uniform float orbitSpeed = 80.0;
uniform float spinSpeed = 240.0;

vec4 axisAngleToQuaternion(vec3 axis, float angle) {
	float halfAngle = radians(angle) / 2.0;
	float s = sin(halfAngle);
	return vec4(axis * s, cos(halfAngle));
}

vec3 rotateVertex(vec3 v , vec4 q) {
	vec3 t = 2.0 * cross(q.xyz, v);
	return v + q.w * t + cross(q.xyz, t);
}


void main() {
	// Apply quaternion rotation to vertex position
	vec3 axis = vec3(0.0, 1.0, 0.0); // Rotate around the Y-axis
	vec4 rotQuaternion = axisAngleToQuaternion(axis, currentTime * spinSpeed);

	vec3 rotPosition = rotateVertex(vertexPosition_modelspace, rotQuaternion);
	vec3 rotNormal = rotateVertex(vertexNormal_modelspace, rotQuaternion);

	// Add orbiting motion
	float orbitAngle = radians(orbitSpeed * currentTime);
	vec3 orbitOffset = vec3(cos(orbitAngle) * orbitRadius, 0.0, sin(orbitAngle) * orbitRadius);
	rotPosition += orbitOffset;

	WorldPos = vec3(M * vec4(rotPosition, 1.0));
	WorldNormal = mat3(transpose(inverse(M))) * rotNormal; // Transform normal
	DebugNormal = vertexNormal_modelspace; // For debugging normal transformation
	gl_Position = P * V * M * vec4(rotPosition, 1.0);
}
