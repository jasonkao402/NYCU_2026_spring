#version 330 core

// Removed location 4 (aBitangent) since we compute it manually
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in vec3 aTangent;

out VS_OUT {
    vec3 FragPos;
    vec2 TexCoords;
    vec3 TangentViewPos;
    vec3 TangentFragPos;
    mat3 TBN;
} vs_out;

// Match the UBO setup from your C++ code
layout (std140) uniform MVP {
    mat4 view;       // Offset 0
    mat4 projection; // Offset 64
};

// Renamed from 'model' to 'M' to match C++ setUniform("M", ...)
uniform mat4 M;
uniform vec3 viewPos; 

void main() {
    // 1. Calculate World Space Position using 'M'
    vs_out.FragPos = vec3(M * vec4(aPos, 1.0));   
    vs_out.TexCoords = aTexCoords;
    
    // 2. Construct the TBN Matrix
    mat3 normalMatrix = transpose(inverse(mat3(M)));
    vec3 T = normalize(normalMatrix * aTangent);
    vec3 N = normalize(normalMatrix * aNormal);
    T = normalize(T - dot(T, N) * N); 
    
    // Calculate Bitangent here instead of passing it from CPU
    vec3 B = cross(N, T); 
    
    vs_out.TBN = mat3(T, B, N);

    // 3. Transform View and Frag positions to Tangent Space
    mat3 invTBN = transpose(vs_out.TBN);
    vs_out.TangentViewPos = invTBN * viewPos;
    vs_out.TangentFragPos = invTBN * vs_out.FragPos;
        
    gl_Position = projection * view * M * vec4(aPos, 1.0);
}