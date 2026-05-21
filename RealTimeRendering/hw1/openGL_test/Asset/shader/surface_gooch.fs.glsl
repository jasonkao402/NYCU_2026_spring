#version 410 core

in vec3 WorldPos;
in vec3 WorldNormal;

out vec4 FragColor;

uniform vec3 SurfaceColor;
uniform float DiffuseWarm;
uniform float DiffuseCool;

uniform vec3 LightPos[3];
uniform vec3 LightColor = vec3(1.0); 
uniform float showNormals;
uniform vec3 ViewPos; 

void main() {
    vec3 N = normalize(WorldNormal);
    vec3 V = normalize(ViewPos - WorldPos);

    // 1. Corrected base constants to match Image 2
    vec3 WarmColor = vec3(0.5, 0.5, 0.0) + DiffuseWarm * SurfaceColor;
    vec3 CoolColor = vec3(0.0, 0.0, 0.55) + DiffuseCool * SurfaceColor;

    vec3 finalColor = vec3(0.0);

    for(int i = 0; i < 3; i++) {
        vec3 L = normalize(LightPos[i] - WorldPos);
        
        // 2. Diffuse mapping (t)
        float NdotL = dot(N, L);
        float t = (NdotL + 1.0) / 2.0; 
        
        // Base diffuse interpolation
        vec3 goochDiffuse = mix(CoolColor, WarmColor, t);

        // 3. Image 2 Specular Math (s)
        vec3 R = reflect(-L, N); // reflect() is mathematically equivalent to 2(n.l)n - l
        float RdotV = dot(R, V);
        
        // s = 100(R.V) - 97. 
        // We clamp it between 0.0 and 1.0 because it's used as an interpolation weight.
        float s = clamp(100.0 * RdotV - 97.0, 0.0, 1.0); 

        // 4. Image 2 Blending (Interpolation, not addition)
        // mix(x, y, a) does: x * (1-a) + y * a
        vec3 lightContribution = mix(goochDiffuse, LightColor, s);

        finalColor += lightContribution;
    }
    finalColor = finalColor / 3.0;
    
    FragColor = (1.0 - showNormals) * vec4(finalColor, 1.0) + showNormals * vec4(N * 0.5 + 0.5, 1.0);
}