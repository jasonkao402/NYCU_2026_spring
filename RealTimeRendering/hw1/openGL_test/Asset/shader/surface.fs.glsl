#version 410 core

in vec3 WorldPos;
in vec3 WorldNormal;
in vec3 DebugNormal;
out vec4 fragColor;

uniform vec3 colors = vec3(0.9059, 0.7725, 0.0); // object color
uniform vec3 lightPos;
uniform vec3 viewPos;

uniform float ambientStrength = 0.1;
uniform float specularStrength = 0.5;
uniform float shininess = 0.5;

void main() {
	// TODO: implement Phong Shading
    // assume ambient light is white
    // Normalize interpolated normal (interpolation may denormalize it)
    vec3 norm = normalize(WorldNormal);
    
    // Light direction (from fragment to light)
    vec3 lightDir = normalize(lightPos - WorldPos);
    
    // View direction (from fragment to camera)
    vec3 viewDir = normalize(viewPos - WorldPos);
    
    // --- Ambient ---
    vec3 ambient = ambientStrength * colors;
    
    // --- Diffuse ---
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * colors;
    
    // --- Specular (Phong reflection model) ---
    vec3 reflectDir = reflect(-lightDir, norm);
    // Angle between view direction and reflection direction
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = specularStrength * spec * vec3(1.0); // Assuming white specular highlight
    
    vec3 result = ambient + diffuse + specular;

    // fragColor = vec4(diff, diff, diff, 1.0);
    // fragColor = vec4(normalize(WorldNormal) * 0.5 + 0.5, 1.0);
    fragColor = vec4(result, 1.0);
}
