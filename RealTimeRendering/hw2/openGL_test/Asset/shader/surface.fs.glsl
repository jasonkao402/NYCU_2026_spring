#version 410 core

in vec3 WorldPos;
in vec3 WorldNormal;

out vec4 FragColor;

uniform vec3 SurfaceColor;
uniform float DiffuseWarm;
uniform float DiffuseCool;

// Dynamic arrays for your 3 point lights
uniform vec3 LightPositions[3];
uniform vec3 LightColor;

// Needed for accurate specular reflection
uniform vec3 ViewPos; 

void main() {
    // Normalize our interpolated vectors
    vec3 N = normalize(WorldNormal);
    vec3 V = normalize(ViewPos - WorldPos);

    // Calculate colors INSIDE main() to fix the initialization error
    vec3 WarmColor = vec3(0.5, 0.5, 0.0) + DiffuseWarm * SurfaceColor;
    vec3 CoolColor = vec3(0.0, 0.0, 0.55) + DiffuseCool * SurfaceColor;

    vec3 finalColor = vec3(0.0);

    // Loop through all 3 lights
    for(int i = 0; i < 3; i++) {
        vec3 L = normalize(LightPositions[i] - WorldPos);
        
        float NdotL = dot(N, L);
        
        // Gooch Diffuse Calculation
        float t = (NdotL + 1.0) / 2.0; // Map from [-1, 1] to [0, 1]
        vec3 goochDiffuse = mix(CoolColor, WarmColor, t);

        // Specular Calculation (Using GLSL's built-in reflect)
        // Note: reflect() expects the incident vector pointing TOWARDS the surface, hence -L
        vec3 R = reflect(-L, N); 
        float specAmount = pow(max(dot(V, R), 0.0), 8.0); // Increased power for tighter highlight
        vec3 specular = specAmount * LightColor;

        // Accumulate light contribution
        finalColor += (goochDiffuse + specular);
    }

    // Average the light result so 3 lights don't blow out the color to pure white
    // finalColor = finalColor / 3.0;

    // FragColor = vec4(finalColor, 1.0);
    FragColor = N * 0.5 + 0.5; // Visualize normals for debugging
}