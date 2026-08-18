#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform CameraBufferObject {
    mat4 view;
    mat4 proj;
} camera;

layout(set = 1, binding = 1) uniform sampler2D baseColorSampler;
layout(set = 1, binding = 2) uniform sampler2D normalSampler;
layout(set = 1, binding = 3) uniform sampler2D metallicRoughnessSampler;
layout(set = 1, binding = 4) uniform sampler2D specularSampler;
layout(set = 1, binding = 5) uniform sampler2D specularColorSampler;

layout(set = 2, binding = 0) uniform ShadowBufferObject {
    mat4 lightViewProjection;
    vec4 lightDirection;
} shadow;
layout(set = 2, binding = 1) uniform sampler2DShadow shadowMap;

layout(set = 3, binding = 0) uniform samplerCube environmentMap;
layout(set = 3, binding = 1) uniform EnvironmentBufferObject {
    vec4 parameters;
    vec4 options;
} environment;
layout(set = 3, binding = 2) uniform samplerCube irradianceMap;
layout(set = 3, binding = 3) uniform samplerCube prefilteredMap;
layout(set = 3, binding = 4) uniform sampler2D brdfLut;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) flat in vec4 fragMaterialParameters;
layout(location = 3) flat in vec4 fragPbrParameters;
layout(location = 4) flat in vec4 fragSpecularColorParameters;
layout(location = 5) in vec3 fragWorldPosition;
layout(location = 6) in vec3 fragNormal;
layout(location = 7) in vec4 fragTangent;
layout(location = 8) in vec4 fragLightSpacePosition;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

float DistributionGGX(vec3 normal, vec3 halfway, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float nDotH = max(dot(normal, halfway), 0.0);
    float denominator = nDotH * nDotH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denominator * denominator, 0.000001);
}

float GeometrySchlickGGX(float nDotDirection, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return nDotDirection /
        max(nDotDirection * (1.0 - k) + k, 0.000001);
}

float GeometrySmith(vec3 normal, vec3 viewDirection, vec3 lightDirection,
                    float roughness) {
    return GeometrySchlickGGX(max(dot(normal, viewDirection), 0.0), roughness) *
           GeometrySchlickGGX(max(dot(normal, lightDirection), 0.0), roughness);
}

vec3 FresnelSchlick(float cosTheta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 FresnelSchlickRoughness(float cosTheta, vec3 f0, float roughness) {
    return f0 + (max(vec3(1.0 - roughness), f0) - f0) *
        pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 rotateAroundY(vec3 direction, float angle) {
    float cosine = cos(angle);
    float sine = sin(angle);
    return vec3(cosine * direction.x - sine * direction.z,
                direction.y,
                sine * direction.x + cosine * direction.z);
}

float ShadowVisibility(vec4 lightSpacePosition, vec3 normal,
                       vec3 lightDirection) {
    vec3 projected = lightSpacePosition.xyz / lightSpacePosition.w;
    vec2 uv = projected.xy * 0.5 + 0.5;
    if (projected.z <= 0.0 || projected.z >= 1.0 ||
        uv.x <= 0.0 || uv.x >= 1.0 ||
        uv.y <= 0.0 || uv.y >= 1.0) {
        return 1.0;
    }

    float normalBias = 1.0 - max(dot(normal, lightDirection), 0.0);
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
    if (!gl_FrontFacing && fragSpecularColorParameters.w < 0.5) {
        discard;
    }

    vec4 baseColor = texture(baseColorSampler, fragTexCoord) *
        vec4(fragColor, 1.0);
    baseColor.a *= fragMaterialParameters.z;

    float alphaMode = fragMaterialParameters.y;
    if (alphaMode < 0.5) {
        baseColor.a = 1.0;
    } else if (alphaMode < 1.5) {
        if (baseColor.a < fragMaterialParameters.x) {
            discard;
        }
        baseColor.a = 1.0;
    }

    vec3 normal = normalize(fragNormal);
    vec3 tangent = normalize(fragTangent.xyz);
    if (!gl_FrontFacing) {
        normal = -normal;
        tangent = -tangent;
    }
    vec3 bitangent = normalize(cross(normal, tangent)) * fragTangent.w;
    vec3 tangentNormal = texture(normalSampler, fragTexCoord).xyz * 2.0 - 1.0;
    tangentNormal.xy *= fragPbrParameters.z;
    normal = normalize(mat3(tangent, bitangent, normal) * tangentNormal);

    vec4 metallicRoughness =
        texture(metallicRoughnessSampler, fragTexCoord);
    float metallic = clamp(
        metallicRoughness.b * fragPbrParameters.x, 0.0, 1.0);
    float roughness = clamp(
        metallicRoughness.g * fragPbrParameters.y, 0.045, 1.0);
    float specularFactor = clamp(
        texture(specularSampler, fragTexCoord).a * fragPbrParameters.w,
        0.0, 1.0);
    vec3 specularColor = texture(specularColorSampler, fragTexCoord).rgb *
        fragSpecularColorParameters.rgb;

    vec3 cameraPosition = inverse(camera.view)[3].xyz;
    vec3 viewDirection = normalize(cameraPosition - fragWorldPosition);
    vec3 lightDirection = normalize(shadow.lightDirection.xyz);
    vec3 halfway = normalize(viewDirection + lightDirection);
    vec3 f0 = mix(vec3(0.04) * specularFactor * specularColor,
                  baseColor.rgb, metallic);

    float distribution =
        DistributionGGX(normal, halfway, roughness);
    float geometry =
        GeometrySmith(normal, viewDirection, lightDirection, roughness);
    vec3 fresnel = FresnelSchlick(
        max(dot(halfway, viewDirection), 0.0), f0);
    vec3 specular = distribution * geometry * fresnel /
        max(4.0 * max(dot(normal, viewDirection), 0.0) *
            max(dot(normal, lightDirection), 0.0), 0.0001);

    vec3 diffuseWeight = (vec3(1.0) - fresnel) * (1.0 - metallic);
    float nDotL = max(dot(normal, lightDirection), 0.0);
    vec3 directRadiance = vec3(3.4, 3.2, 3.0) * environment.options.z;
    vec3 directLighting =
        (diffuseWeight * baseColor.rgb / PI + specular) *
        directRadiance * nDotL * ShadowVisibility(
            fragLightSpacePosition, normal, lightDirection);
    vec3 ambientLighting;
    if (environment.parameters.w > 0.5) {
        vec3 environmentNormal = rotateAroundY(
            normal, environment.parameters.z);
        vec3 reflectedDirection = rotateAroundY(
            reflect(-viewDirection, normal), environment.parameters.z);
        float nDotV = max(dot(normal, viewDirection), 0.0);
        vec3 environmentFresnel = FresnelSchlickRoughness(
            nDotV, f0, roughness);
        vec3 diffuseWeightEnvironment =
            (vec3(1.0) - environmentFresnel) * (1.0 - metallic);

        vec3 irradiance = texture(irradianceMap, environmentNormal).rgb;
        vec3 environmentDiffuse = irradiance * baseColor.rgb / PI;

        vec3 prefiltered = textureLod(
            prefilteredMap, reflectedDirection,
            roughness * environment.options.y).rgb;
        vec2 brdf = texture(brdfLut, vec2(nDotV, roughness)).rg;
        vec3 environmentSpecular = prefiltered *
            (environmentFresnel * brdf.x + brdf.y);
        ambientLighting =
            (diffuseWeightEnvironment * environmentDiffuse +
             environmentSpecular) * environment.parameters.y;
    } else {
        // Preserve the old neutral fallback when a scene has no HDR.
        ambientLighting =
            baseColor.rgb * 0.22 * (1.0 - metallic * 0.5);
    }
    vec3 finalColor = ambientLighting + directLighting;

    // Keep the PBR result bounded for the display. The sRGB swapchain performs
    // the linear-to-sRGB encoding when this value is written to the attachment.
    vec3 toneMapped = finalColor / (finalColor + vec3(1.0));
    outColor = vec4(max(toneMapped, vec3(0.0)), baseColor.a);
}
