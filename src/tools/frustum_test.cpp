#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>

#include "Frustum.h"

int main() {
    const glm::mat4 view = glm::lookAt(
        glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 projection = glm::perspective(
        glm::radians(60.0f), 1.0f, 0.1f, 100.0f);
    projection[1][1] *= -1.0f;
    const Frustum frustum(projection * view);

    const bool inside = frustum.IntersectsAabb(
        glm::vec3(-0.5f), glm::vec3(0.5f));
    const bool behind = frustum.IntersectsAabb(
        glm::vec3(-0.5f, -0.5f, 6.0f),
        glm::vec3(0.5f, 0.5f, 7.0f));
    const bool side = frustum.IntersectsAabb(
        glm::vec3(20.0f, -0.5f, -0.5f),
        glm::vec3(21.0f, 0.5f, 0.5f));
    const bool intersect = frustum.IntersectsAabb(
        glm::vec3(-4.0f, -0.5f, -0.5f),
        glm::vec3(0.0f, 0.5f, 0.5f));

    const bool passed = inside && !behind && !side && intersect;
    std::cout << "FRUSTUM_TEST " << (passed ? "PASS" : "FAIL")
              << " inside=" << inside
              << " behind=" << behind
              << " side=" << side
              << " intersect=" << intersect << '\n';
    return passed ? 0 : 1;
}
