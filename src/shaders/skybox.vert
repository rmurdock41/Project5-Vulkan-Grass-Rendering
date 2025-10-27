#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform CameraBufferObject {
    mat4 view;
    mat4 proj;
} camera;

layout(location = 0) in vec3 inPosition;

layout(location = 0) out vec3 fragTexCoord;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    // Remove translation from view matrix (only rotation)
    mat4 viewNoTranslation = mat4(mat3(camera.view));
    
    vec4 pos = camera.proj * viewNoTranslation * vec4(inPosition, 1.0);
    
    // Set z = w so depth is always 1.0 (farthest)
    gl_Position = pos.xyww;
    
    fragTexCoord = inPosition;
}