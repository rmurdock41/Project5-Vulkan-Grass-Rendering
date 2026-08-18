#pragma once

#include <array>

#include <glm/glm.hpp>

class Frustum {
public:
    explicit Frustum(const glm::mat4& viewProjection);

    bool IntersectsAabb(const glm::vec3& boundsMin,
                        const glm::vec3& boundsMax) const;

private:
    std::array<glm::vec4, 6> planes;
};
