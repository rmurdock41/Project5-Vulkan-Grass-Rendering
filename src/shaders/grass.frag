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
    
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));  // Sun direction
    vec3 normal = normalize(fragNormal);
    
    // Diffuse lighting 
    float NdotL = dot(normal, lightDir);
    float diffuse = (NdotL + 0.5) / 1.5;  // Wrap lighting
    diffuse = clamp(diffuse, 0.0, 1.0);
    
    // Ambient light (base lighting when no direct sun)
    vec3 ambient = grassColor * 0.3;
    

    vec3 diffuseColor = grassColor * diffuse * 0.7;
    
    vec3 viewDir = normalize(vec3(0.0, 1.0, 0.0));
    
    // Rim effect: edges perpendicular to view are brighter
    float rim = 1.0 - abs(dot(normal, viewDir));
    rim = pow(rim, 3.0);  // Sharp falloff
    
    vec3 rimColor = vec3(0.8, 1.0, 0.6) * rim * 0.3;  //  rimLight
    
    vec3 finalColor = ambient + diffuseColor + rimColor;
    
    outColor = vec4(finalColor, 1.0);
}
