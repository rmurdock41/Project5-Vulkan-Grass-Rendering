#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(quads, equal_spacing, ccw) in;

layout(set = 0, binding = 0) uniform CameraBufferObject {
    mat4 view;
    mat4 proj;
} camera;

// TODO: Declare tessellation evaluation shader inputs and outputs

layout(location = 0) in vec4 in_v0[];
layout(location = 1) in vec4 in_v1[];
layout(location = 2) in vec4 in_v2[];
layout(location = 3) in vec4 in_up[];


layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec2 fragUV;

void main() {
    float u = gl_TessCoord.x;
    float v = gl_TessCoord.y;
    
	// TODO: Use u and v to parameterize along the grass blade and output positions for each vertex of the grass blade

    vec3 v0 = in_v0[0].xyz;
    vec3 v1 = in_v1[0].xyz;
    vec3 v2 = in_v2[0].xyz;
    vec3 up = in_up[0].xyz;
    
    float orientation = in_v0[0].w;
    float height = in_v1[0].w;
    float width = in_v2[0].w;
    
    vec3 t1 = vec3(sin(orientation), 0.0, cos(orientation));
    
    float t = v;
    vec3 a = v0 + t * (v1 - v0);
    vec3 b = v1 + t * (v2 - v1);
    vec3 c = a + t * (b - a);
    

    float currentWidth = width * (1.0 - v);

    vec3 c0 = c - currentWidth * t1;  
    vec3 c1 = c + currentWidth * t1;  
    
    vec3 worldPos = mix(c0, c1, u);
    
    
    vec3 normal = normalize(cross(t1, normalize(v2 - v0)));
    fragNormal = normal;
    
    fragUV = vec2(u, v);
    
    gl_Position = camera.proj * camera.view * vec4(worldPos, 1.0);
}
