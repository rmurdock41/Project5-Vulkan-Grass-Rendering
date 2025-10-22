#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform CameraBufferObject {
    mat4 view;
    mat4 proj;
} camera;

// TODO: Declare fragment shader inputs

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

void main() {
    // TODO: Compute fragment color

    vec3 grassColorBottom = vec3(0.1, 0.4, 0.1);
    vec3 grassColorTop = vec3(0.3, 0.8, 0.3);
    
    vec3 grassColor = mix(grassColorBottom, grassColorTop, fragUV.y);
    
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    float diff = max(dot(fragNormal, lightDir), 0.0);
    diff = diff * 0.5 + 0.5;
    
    vec3 finalColor = grassColor * diff;
    
    outColor = vec4(finalColor, 1.0);

}
