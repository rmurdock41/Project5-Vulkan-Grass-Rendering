#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShadowBufferObject {
    mat4 lightViewProjection;
    vec4 lightDirection;
} shadow;

layout(set = 1, binding = 0) uniform ModelBufferObject {
    mat4 model;
    vec4 materialParameters;
    vec4 pbrParameters;
    vec4 specularColorParameters;
};

layout(location = 0) in vec3 inPosition;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) flat out vec4 fragMaterialParameters;

void main() {
    gl_Position = shadow.lightViewProjection * model *
        vec4(inPosition, 1.0);
    fragTexCoord = inTexCoord;
    fragMaterialParameters = materialParameters;
}
