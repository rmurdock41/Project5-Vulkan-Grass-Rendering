#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform CameraBufferObject {
    mat4 view;
	mat4 proj;
} camera;

layout(set = 1, binding = 0) uniform ModelBufferObject {
    mat4 model;
    vec4 materialParameters;
    vec4 pbrParameters;
    vec4 specularColorParameters;
};

layout(set = 2, binding = 0) uniform ShadowBufferObject {
    mat4 lightViewProjection;
    vec4 lightDirection;
} shadow;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inNormal;
layout(location = 4) in vec4 inTangent;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) flat out vec4 fragMaterialParameters;
layout(location = 3) flat out vec4 fragPbrParameters;
layout(location = 4) flat out vec4 fragSpecularColorParameters;
layout(location = 5) out vec3 fragWorldPosition;
layout(location = 6) out vec3 fragNormal;
layout(location = 7) out vec4 fragTangent;
layout(location = 8) out vec4 fragLightSpacePosition;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    vec4 worldPosition = model * vec4(inPosition, 1.0);
    mat3 normalMatrix = transpose(inverse(mat3(model)));
    vec3 worldNormal = normalize(normalMatrix * inNormal);
    vec3 worldTangent = mat3(model) * inTangent.xyz;
    worldTangent = normalize(worldTangent -
        worldNormal * dot(worldNormal, worldTangent));
    float transformSign = determinant(mat3(model)) < 0.0 ? -1.0 : 1.0;

    gl_Position = camera.proj * camera.view * worldPosition;
    fragColor = inColor;
    fragTexCoord = inTexCoord;
    fragMaterialParameters = materialParameters;
    fragPbrParameters = pbrParameters;
    fragSpecularColorParameters = specularColorParameters;
    fragWorldPosition = worldPosition.xyz;
    fragNormal = worldNormal;
    fragTangent = vec4(worldTangent, inTangent.w * transformSign);
    fragLightSpacePosition = shadow.lightViewProjection * worldPosition;
}
