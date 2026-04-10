#version 410 core

in vec3 WorldPos;
in vec3 WorldNormal;
in vec3 DebugNormal;
out vec4 fragColor;

uniform vec3 colors = vec3(0.9059, 0.7725, 0.0); // object color
uniform vec3 lightPos;
uniform vec3 viewPos;

uniform float ambientStrength = 0.333;
uniform float specularStrength = 0.5;
uniform float shininess = 0.5;
const float levels = 5;

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main() {
	vec3 norm = normalize(WorldNormal);
    vec3 lightDir = normalize(lightPos - WorldPos);
    vec3 viewDir = normalize(viewPos - WorldPos);
    
    // --- 1. Ambient ---
    // Keep this low so the dark areas stay flat
    vec3 ambient = ambientStrength * colors;
    
    // --- 2. Toon Diffuse ---
    float smoothDiff = max(dot(norm, lightDir), 0.0);
    
    // The Magic Math: Multiply by levels, round down, then divide by levels.
    // E.g., if smoothDiff is 0.6 and levels is 4: 0.6 * 4 = 2.4 -> floor(2.4) = 2.0 -> 2.0 / 4.0 = 0.5.
    float toonDiff = floor(smoothDiff * levels) / levels;
    vec3 diffuse = toonDiff * colors;
    
    // --- 3. Toon Specular with Fresnel ---
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = max(dot(viewDir, reflectDir), 0.0);
    
    // Hard cutoff: If the specular reflection is very strong, make it 100% white. Otherwise, 0%.
    float toonSpec = 0.0;
    if (spec > 0.95) { // 0.95 is the threshold. Adjust to make the highlight bigger/smaller
        toonSpec = 1.0; 
    }
    
    // Fresnel effect
    vec3 F0 = vec3(0.04); // Base reflectivity for non-metals
    F0 = mix(F0, colors, 0.5); // Adjust for material properties
    vec3 fresnel = fresnelSchlick(max(dot(viewDir, norm), 0.0), F0);
    vec3 specular = toonSpec * fresnel; 
    
    // --- Final Output ---
    vec3 result = ambient + diffuse + specular;
    fragColor = vec4(result, 1.0);
}
