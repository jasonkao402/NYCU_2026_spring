#version 410 core

out vec3 color;

in vec3 WorldNormal;
in vec3 WorldPos;
in float NdotL;

uniform vec3 SurfaceColor;
uniform float DiffuseWarm;
uniform float DiffuseCool;
uniform vec3 WarmColor = vec3(0.5, 0.5, 0.0) + DiffuseWarm * SurfaceColor; // Yellowish warm color
uniform vec3 CoolColor = vec3(0.0, 0.0, 0.55) + DiffuseCool * SurfaceColor; // Blueish cool color
uniform vec3 LightColor = vec3(1.0); // White light

void main() {
    float t = (NdotL + 1.0) / 2.0; // Map from [-1, 1] to [0, 1]
	float r = 2 * NdotL * N - L; // Reflection vector
	float s = max(dot(r, L), 0.0); // Specular term for fresnel effect
	s = pow(s, 5.0); // Sharpen the specular highlight

    // vec3 GoochColor = ; 
    color = mix(CoolColor, WarmColor, t);
}