#version 410 core

layout(location = 0) in vec3 vertexPosition_modelspace;

layout(std140) uniform MVP {
	mat4 V; // 0
	mat4 P; // 64
};

uniform mat4 M;
uniform float rotationAngle;

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
	// TODO: apply quaternion rotation to vertex position
	vec3 axis = vec3(0.0, 1.0, 0.0); // Rotate around the Y-axis
	vec4 rotQuaternion = axisAngleToQuaternion(axis, rotationAngle);
	vec3 rotPosition = rotateVertex(vertexPosition_modelspace, rotQuaternion);
	gl_Position = P * V * M * vec4(rotPosition, 1.0);
}
