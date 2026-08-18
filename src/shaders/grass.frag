#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform CameraBufferObject {
    mat4 view;
    mat4 proj;
} camera;

layout(set = 2, binding = 0) uniform ShadowBufferObject {
    mat4 lightViewProjection;
    vec4 lightDirection;
} shadow;
layout(set = 2, binding = 1) uniform sampler2DShadow shadowMap;

layout(set = 1, binding = 0) uniform GrassAppearance {
    mat4 model;
    vec4 bottomColor;
    vec4 topColor;
    vec4 rimColor;
} grassAppearance;

// TODO: Declare fragment shader inputs

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec3 fragWorldPosition;

layout(location = 0) out vec4 outColor;

vec3 srgbToLinear(vec3 color) {
    bvec3 cutoff = lessThanEqual(color, vec3(0.04045));
    vec3 lower = color / 12.92;
    vec3 higher = pow((color + 0.055) / 1.055, vec3(2.4));
    return mix(higher, lower, cutoff);
}

float ShadowVisibility(vec3 worldPosition, vec3 normal,
                       vec3 lightDirection) {
    vec4 lightSpace = shadow.lightViewProjection * vec4(worldPosition, 1.0);
    vec3 projected = lightSpace.xyz / lightSpace.w;
    vec2 uv = projected.xy * 0.5 + 0.5;
    if (projected.z <= 0.0 || projected.z >= 1.0 ||
        uv.x <= 0.0 || uv.x >= 1.0 ||
        uv.y <= 0.0 || uv.y >= 1.0) {
        return 1.0;
    }

    float normalBias = 1.0 - abs(dot(normal, lightDirection));
    float bias = max(0.00045, 0.0025 * normalBias);
    vec2 texelSize = 1.0 / vec2(textureSize(shadowMap, 0));
    float visibility = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            visibility += texture(
                shadowMap,
                vec3(uv + vec2(x, y) * texelSize,
                     projected.z - bias));
        }
    }
    return visibility / 9.0;
}

void main() {
    // TODO: Compute fragment color

    vec3 grassColorBottom = srgbToLinear(grassAppearance.bottomColor.rgb);
    vec3 grassColorTop = srgbToLinear(grassAppearance.topColor.rgb);
    
    vec3 grassColor = mix(grassColorBottom, grassColorTop, fragUV.y);
    
    vec3 lightDir = normalize(shadow.lightDirection.xyz);
    vec3 normal = normalize(fragNormal);
    float visibility = ShadowVisibility(
        fragWorldPosition, normal, lightDir);
    
    // Diffuse lighting 
    float NdotL = dot(normal, lightDir);
    float diffuse = (NdotL + 0.5) / 1.5;  // Wrap lighting
    diffuse = clamp(diffuse, 0.0, 1.0);
    
    // Ambient light (base lighting when no direct sun)
    vec3 ambient = grassColor * 0.3;
    

    vec3 diffuseColor = grassColor * diffuse * 0.7 * visibility;
    
    vec3 viewDir = normalize(vec3(0.0, 1.0, 0.0));
    
    // Rim effect: edges perpendicular to view are brighter
    float rim = 1.0 - abs(dot(normal, viewDir));
    rim = pow(rim, 3.0);  // Sharp falloff
    
    vec3 rimColor = srgbToLinear(grassAppearance.rimColor.rgb) * rim * 0.3 *
        mix(0.4, 1.0, visibility);
    
    vec3 finalColor = ambient + diffuseColor + rimColor;
    
    outColor = vec4(finalColor, 1.0);
}
