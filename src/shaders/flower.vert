#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform CameraBufferObject {
    mat4 view;
    mat4 proj;
} camera;

layout(push_constant) uniform FlowerRenderSettings {
    float heightScale;
} flowerSettings;

layout(location = 0) in vec3 localPosition;
layout(location = 1) in vec3 localNormal;
layout(location = 2) in float materialId;
layout(location = 3) in vec4 bladeV0;
layout(location = 4) in vec4 bladeV1;
layout(location = 5) in vec4 bladeV2;
layout(location = 6) in vec4 bladeUp;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec3 fragWorldPosition;
layout(location = 2) flat out float fragMaterialId;

float stableRandom(vec2 seed) {
    return fract(sin(dot(seed, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    vec3 v0 = bladeV0.xyz;
    vec3 v1 = bladeV1.xyz;
    vec3 v2 = bladeV2.xyz;
    float t = clamp(localPosition.y, 0.0, 1.0);
    vec3 a = mix(v0, v1, t);
    vec3 b = mix(v1, v2, t);
    vec3 centre = mix(a, b, t);
    centre = v0 + (centre - v0) * flowerSettings.heightScale;

    // Use the whole stem for orientation. A per-instance horizontal reference
    // is projected onto its perpendicular plane, avoiding the old Y/X
    // reference-axis threshold and its abrupt 90-degree basis changes.
    vec3 wholeStem = v2 - v0;
    vec3 stemUp = dot(wholeStem, wholeStem) > 0.000001
        ? normalize(wholeStem)
        : normalize(bladeUp.xyz);
    float referenceAngle = bladeV0.w;
    vec3 fixedReference = vec3(cos(referenceAngle), 0.0,
                               sin(referenceAngle));
    vec3 stemRight = normalize(fixedReference -
                               stemUp * dot(fixedReference, stemUp));
    vec3 stemForward = normalize(cross(stemUp, stemRight));

    // Each blossom receives a deterministic random lean and azimuth.  The
    // random seed is its world position, so the result never changes when
    // the camera moves.
    float randomLean = stableRandom(v0.xz + vec2(5.31, 17.73));
    float randomAzimuth = stableRandom(v0.zx + vec2(31.17, 8.91)) *
        6.28318530718;
    float randomSpin = stableRandom(v0.xz + vec2(73.41, 41.29)) *
        6.28318530718;
    float leanAngle = radians(mix(7.0, 28.0, randomLean));
    vec3 leanDirection = cos(randomAzimuth) * stemRight +
        sin(randomAzimuth) * stemForward;
    vec3 flowerUp = normalize(cos(leanAngle) * stemUp +
                              sin(leanAngle) * leanDirection);

    // Analytically rotate the frame through the lean angle. Unlike projecting
    // an arbitrary axis and falling back to another one, this remains
    // continuous for the entire wind motion.
    vec3 leanTangent = normalize(-sin(leanAngle) * stemUp +
                                 cos(leanAngle) * leanDirection);
    vec3 leanSide = normalize(cross(flowerUp, leanTangent));
    vec3 flowerRight = cos(randomSpin) * leanTangent +
        sin(randomSpin) * leanSide;
    vec3 flowerForward = -sin(randomSpin) * leanTangent +
        cos(randomSpin) * leanSide;

    bool isBlossom = materialId > 0.5;
    vec3 right = isBlossom ? flowerRight : stemRight;
    vec3 forward = isBlossom ? flowerForward : stemForward;
    vec3 up = isBlossom ? flowerUp : stemUp;
    // Keep the blossom visibly smaller than the surrounding architectural
    // detail; stem height is unchanged, only its radial profile is reduced.
    float lateralScale = bladeV1.w * 0.70;
    vec3 lateral = (right * localPosition.x +
                    forward * localPosition.z) * lateralScale;

    // Stable per-instance variation avoids a visibly cloned flower field.
    float variation = 0.88 + 0.22 * stableRandom(v0.xz);
    vec3 worldPosition = centre + lateral * variation;
    mat3 orientation = mat3(right, up, forward);
    fragNormal = normalize(orientation * localNormal);
    fragWorldPosition = worldPosition;
    fragMaterialId = materialId;
    gl_Position = camera.proj * camera.view * vec4(worldPosition, 1.0);
}
