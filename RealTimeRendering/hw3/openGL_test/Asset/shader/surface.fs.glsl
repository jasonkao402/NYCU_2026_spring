#version 330 core
out vec4 FragColor;

in VS_OUT {
    vec3 FragPos;
    vec2 TexCoords;
    vec3 TangentViewPos;
    vec3 TangentFragPos;
    mat3 TBN;
} fs_in;

// Textures
uniform sampler2D diffuseMap;
uniform sampler2D normalMap;
uniform sampler2D depthMap;

// POM Settings
uniform float depthFactor; // Typically around 0.05 to 0.1

// Lighting Setup (Arbitrary Point Lights)
struct PointLight {
    vec3 position;
    vec3 color;
    float constant;
    float linear;
    float quadratic;
};

#define MAX_POINT_LIGHTS 10
uniform PointLight pointLights[MAX_POINT_LIGHTS];
uniform int numPointLights;
uniform vec3 viewPos; // World space view position
uniform int renderMode;

// --- PARALLAX OCCLUSION MAPPING FUNCTION ---
vec2 ParallaxMapping(vec2 texCoords, vec3 viewDir) { 
    // Determine number of layers based on viewing angle
    const float minLayers = 8.0;
    const float maxLayers = 32.0;
    float numLayers = mix(maxLayers, minLayers, abs(dot(vec3(0.0, 0.0, 1.0), viewDir)));  

    // Calculate layer depth and texture coordinate shift per layer
    float layerDepth = 1.0 / numLayers;
    float currentLayerDepth = 0.0;
    vec2 P = viewDir.xy / viewDir.z * depthFactor; 
    vec2 deltaTexCoords = P / numLayers;
  
    // Initial variables
    vec2  currentTexCoords     = texCoords;
    float currentDepthMapValue = texture(depthMap, currentTexCoords).r;
      
    // Ray-march through the depth map
    while(currentLayerDepth < currentDepthMapValue) {
        currentTexCoords -= deltaTexCoords;
        currentDepthMapValue = texture(depthMap, currentTexCoords).r;  
        currentLayerDepth += layerDepth;  
    }
    
    // Interpolate between the before-intersection and after-intersection for accuracy
    vec2 prevTexCoords = currentTexCoords + deltaTexCoords;
    float afterDepth  = currentDepthMapValue - currentLayerDepth;
    float beforeDepth = texture(depthMap, prevTexCoords).r - currentLayerDepth + layerDepth;
 
    float weight = afterDepth / (afterDepth - beforeDepth);
    vec2 finalTexCoords = prevTexCoords * weight + currentTexCoords * (1.0 - weight);

    return finalTexCoords;
}

void main() {
    // 1. Calculate Tangent-Space View Direction
    vec3 tangentViewDir = normalize(fs_in.TangentViewPos - fs_in.TangentFragPos);
    
    // 2. Perform Parallax Occlusion Mapping
    vec2 texCoords = ParallaxMapping(fs_in.TexCoords, tangentViewDir);
    
    // Discard fragments at the edges to prevent texture bleeding
    if(texCoords.x > 1.0 || texCoords.y > 1.0 || texCoords.x < 0.0 || texCoords.y < 0.0)
        discard;

	// --- DEPTH MODE (2) ---
    if (renderMode == 2) {
        // Sample the depth map at the parallax-shifted UVs
        float depthValue = texture(depthMap, texCoords).r;
        // Output as grayscale
        FragColor = vec4(vec3(depthValue), 1.0);
        return; 
    }

    // --- NORMAL MODE (1) ---
    vec3 tangentNormal = texture(normalMap, texCoords).rgb;
    
    if (renderMode == 1) {
        FragColor = vec4(tangentNormal, 1.0);
        return;
    }
	// --- LIT MODE (0) ---
    // 3. Sample Textures using the new POM TexCoords
    vec3 albedo = texture(diffuseMap, texCoords).rgb;
    
    // 4. Normal Mapping -> Convert Tangent Normal to World Space Normal
    tangentNormal = tangentNormal * 2.0 - 1.0; // Map from [0,1] to [-1,1]
    vec3 worldNormal = normalize(fs_in.TBN * tangentNormal);
    
    // 5. Phong Lighting Loop (World Space)
    vec3 worldViewDir = normalize(viewPos - fs_in.FragPos);
    vec3 result = albedo * 0.1; // Ambient base
    
    int actualLights = min(numPointLights, MAX_POINT_LIGHTS);
    for(int i = 0; i < actualLights; i++) {
        // Light direction and distance
        vec3 lightDir = normalize(pointLights[i].position - fs_in.FragPos);
        float distance = length(pointLights[i].position - fs_in.FragPos);
        
        // Attenuation
        float attenuation = 1.0 / (pointLights[i].constant + 
                                   pointLights[i].linear * distance + 
                                   pointLights[i].quadratic * (distance * distance));
        
        // Diffuse
        float diff = max(dot(worldNormal, lightDir), 0.0);
        vec3 diffuse = diff * pointLights[i].color * albedo;
        
        // Specular (Phong)
        vec3 reflectDir = reflect(-lightDir, worldNormal);
        float spec = pow(max(dot(worldViewDir, reflectDir), 0.0), 32.0); // 32.0 is shininess
        vec3 specular = spec * pointLights[i].color; // Add specular map sampling here if you have one
        
        // Combine
        result += (diffuse + specular) * attenuation;
    }
    
    // Gamma correction is recommended here if not outputting to an sRGB framebuffer
    FragColor = vec4(result, 1.0);
}