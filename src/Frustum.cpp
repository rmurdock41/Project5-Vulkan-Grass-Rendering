#include "Frustum.h"

#include <cmath>

namespace {
glm::vec4 MatrixRow(const glm::mat4& matrix, int row) {
    return glm::vec4(matrix[0][row], matrix[1][row],
                     matrix[2][row], matrix[3][row]);
}

glm::vec4 NormalizePlane(const glm::vec4& plane) {
    const float normalLength = glm::length(glm::vec3(plane));
    if (normalLength <= 0.0f) {
        return plane;
    }
    return plane / normalLength;
}
}

Frustum::Frustum(const glm::mat4& viewProjection) {
    const glm::vec4 row0 = MatrixRow(viewProjection, 0);
    const glm::vec4 row1 = MatrixRow(viewProjection, 1);
    const glm::vec4 row2 = MatrixRow(viewProjection, 2);
    const glm::vec4 row3 = MatrixRow(viewProjection, 3);

    // Vulkan's clip-space depth range is [0, w]. The near plane is therefore
    // row2 rather than the OpenGL-style row3 + row2.
    planes[0] = NormalizePlane(row3 + row0); // left
    planes[1] = NormalizePlane(row3 - row0); // right
    planes[2] = NormalizePlane(row3 + row1); // bottom
    planes[3] = NormalizePlane(row3 - row1); // top
    planes[4] = NormalizePlane(row2);        // near
    planes[5] = NormalizePlane(row3 - row2); // far
}

bool Frustum::IntersectsAabb(const glm::vec3& boundsMin,
                             const glm::vec3& boundsMax) const {
    const glm::vec3 center = (boundsMin + boundsMax) * 0.5f;
    const glm::vec3 extent = (boundsMax - boundsMin) * 0.5f;

    for (std::size_t index = 0; index < planes.size(); ++index) {
        const glm::vec3 normal(planes[index]);
        const float distance =
            glm::dot(normal, center) + planes[index].w;
        const float projectedRadius =
            glm::dot(glm::abs(normal), extent);
        if (distance + projectedRadius < 0.0f) {
            return false;
        }
    }
    return true;
}
