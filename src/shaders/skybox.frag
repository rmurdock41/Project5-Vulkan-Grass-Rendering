#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragTexCoord;
layout(location = 0) out vec4 outColor;

void main() {
    vec3 dir = normalize(fragTexCoord);
    
    // Sky Gradient
    float height = dir.y;  // -1 (bottom) to 1 (top)
    
    // Colors
    vec3 skyColorTop = vec3(0.1, 0.3, 0.8);        // Dark blue at zenith
    vec3 skyColorHorizon = vec3(0.6, 0.8, 1.0);    // Light blue at horizon
    vec3 groundColor = vec3(0.4, 0.35, 0.3);       // Brown below horizon
    
    vec3 skyColor;
    if (height > 0.0) {
        // Above horizon - gradient from horizon to top
        float t = pow(height, 0.7);  // Non-linear for better look
        skyColor = mix(skyColorHorizon, skyColorTop, t);
    } else {
        // Below horizon - fade to ground color
        float t = pow(-height, 0.5);
        skyColor = mix(skyColorHorizon, groundColor, t);
    }
    
    // Sun
    vec3 sunDir = normalize(vec3(0.5, 0.8, 0.3));  // Sun position
    float sun = dot(dir, sunDir);
    sun = smoothstep(0.995, 0.999, sun);  // Sharp sun disk
    
    // Sun glow
    float sunGlow = dot(dir, sunDir);
    sunGlow = max(sunGlow, 0.0);
    sunGlow = pow(sunGlow, 8.0) * 0.3;  // Soft glow around sun
    
    vec3 sunColor = vec3(1.0, 0.9, 0.7);
    skyColor += sunColor * (sun + sunGlow);
    
    // Clouds
    // Simple procedural clouds using direction
    float cloudPattern = sin(dir.x * 10.0 + dir.z * 8.0) * 
                         cos(dir.x * 8.0 - dir.z * 10.0);
    cloudPattern = cloudPattern * 0.5 + 0.5;  // Remap to 0-1
    
    // Only show clouds in upper sky
    float cloudMask = smoothstep(0.0, 0.3, height) * 
                      smoothstep(1.0, 0.5, height);
    
    cloudPattern = pow(cloudPattern, 3.0);  // Make clouds puffier
    vec3 cloudColor = vec3(1.0, 1.0, 1.0);
    skyColor = mix(skyColor, cloudColor, cloudPattern * cloudMask * 0.5);
    
    
    outColor = vec4(skyColor, 1.0);
}