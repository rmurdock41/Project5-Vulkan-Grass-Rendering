#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragTexCoord;
layout(location = 0) out vec4 outColor;

layout(set = 1, binding = 0) uniform samplerCube environmentMap;
layout(set = 1, binding = 1) uniform EnvironmentBufferObject {
    vec4 parameters;
    vec4 options;
} environment;

vec3 srgbToLinear(vec3 color) {
    bvec3 cutoff = lessThanEqual(color, vec3(0.04045));
    vec3 lower = color / 12.92;
    vec3 higher = pow((color + 0.055) / 1.055, vec3(2.4));
    return mix(higher, lower, cutoff);
}

vec3 rotateAroundY(vec3 direction, float angle) {
    float cosine = cos(angle);
    float sine = sin(angle);
    return vec3(cosine * direction.x - sine * direction.z,
                direction.y,
                sine * direction.x + cosine * direction.z);
}

vec3 toneMap(vec3 color) {
    return color / (color + vec3(1.0));
}

void main() {
    vec3 dir = normalize(fragTexCoord);

    if (environment.parameters.w > 0.5) {
        if (environment.options.x < 0.5) {
            outColor = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }
        vec3 rotatedDirection = rotateAroundY(
            dir, environment.parameters.z);
        vec3 hdrColor = texture(environmentMap, rotatedDirection).rgb *
            environment.parameters.x;
        outColor = vec4(toneMap(max(hdrColor, vec3(0.0))), 1.0);
        return;
    }
    
    // Sky Gradient
    float height = dir.y;  // -1 (bottom) to 1 (top)
    
    // Colors
    vec3 skyColorTop = srgbToLinear(vec3(0.1, 0.3, 0.8));        // Dark blue at zenith
    vec3 skyColorHorizon = srgbToLinear(vec3(0.6, 0.8, 1.0));    // Light blue at horizon
    vec3 groundColor = srgbToLinear(vec3(0.4, 0.35, 0.3));       // Brown below horizon
    
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
    
    vec3 sunColor = srgbToLinear(vec3(1.0, 0.9, 0.7));
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
    vec3 cloudColor = srgbToLinear(vec3(1.0, 1.0, 1.0));
    skyColor = mix(skyColor, cloudColor, cloudPattern * cloudMask * 0.5);
    
    
    outColor = vec4(skyColor, 1.0);
}
