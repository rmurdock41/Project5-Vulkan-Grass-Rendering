#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 1, binding = 1) uniform sampler2D baseColorSampler;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) flat in vec4 fragMaterialParameters;

void main() {
    float alphaMode = fragMaterialParameters.y;
    if (alphaMode > 0.5) {
        float alpha = texture(baseColorSampler, fragTexCoord).a *
            fragMaterialParameters.z;
        float cutoff = alphaMode < 1.5
            ? fragMaterialParameters.x
            : 0.20;
        if (alpha < cutoff) {
            discard;
        }
    }
}
