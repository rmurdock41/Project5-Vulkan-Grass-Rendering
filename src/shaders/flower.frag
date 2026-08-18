#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 2, binding = 0) uniform ShadowBufferObject {
    mat4 lightViewProjection;
    vec4 lightDirection;
} shadow;
layout(set = 2, binding = 1) uniform sampler2DShadow shadowMap;

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragWorldPosition;
layout(location = 2) flat in float fragMaterialId;

layout(location = 0) out vec4 outColor;

vec3 srgbToLinear(vec3 color) {
    bvec3 cutoff = lessThanEqual(color, vec3(0.04045));
    vec3 lower = color / 12.92;
    vec3 higher = pow((color + 0.055) / 1.055, vec3(2.4));
    return mix(higher, lower, cutoff);
}

float shadowVisibility(vec3 normal, vec3 lightDirection) {
    vec4 lightSpace = shadow.lightViewProjection *
        vec4(fragWorldPosition, 1.0);
    vec3 projected = lightSpace.xyz / lightSpace.w;
    vec2 uv = projected.xy * 0.5 + 0.5;
    if (projected.z <= 0.0 || projected.z >= 1.0 ||
        uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0) {
        return 1.0;
    }
    float bias = max(0.00045,
        0.0025 * (1.0 - abs(dot(normal, lightDirection))));
    vec2 texel = 1.0 / vec2(textureSize(shadowMap, 0));
    float result = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            result += texture(shadowMap,
                vec3(uv + vec2(x, y) * texel, projected.z - bias));
        }
    }
    return result / 9.0;
}

void main() {
    vec3 stem = srgbToLinear(vec3(0.12, 0.18, 0.13));
    vec3 petal = srgbToLinear(vec3(0.82, 0.86, 0.84));
    vec3 centre = srgbToLinear(vec3(0.60, 0.47, 0.22));
    vec3 baseColor = fragMaterialId < 0.5 ? stem :
        (fragMaterialId < 1.5 ? petal : centre);

    vec3 normal = normalize(fragNormal);
    if (!gl_FrontFacing) {
        normal = -normal;
    }
    vec3 lightDirection = normalize(shadow.lightDirection.xyz);
    float visibility = shadowVisibility(normal, lightDirection);
    float diffuse = clamp((dot(normal, lightDirection) + 0.45) / 1.45,
                          0.0, 1.0);
    vec3 color = baseColor * (0.34 + 0.78 * diffuse * visibility);
    outColor = vec4(color, 1.0);
}
