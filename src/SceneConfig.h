#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace SceneConfig {

struct ModelEntry {
    std::string path;
    glm::vec3 position = glm::vec3(0.0f);
    glm::vec3 rotationDegrees = glm::vec3(0.0f);
    glm::vec3 scale = glm::vec3(1.0f);
};

struct EnvironmentEntry {
    std::string path;
    float backgroundIntensity = 1.0f;
    float lightingIntensity = 1.0f;
    float directLightingIntensity = 1.0f;
    float rotationDegrees = 0.0f;
    bool visible = true;
    bool enabled = false;
};

struct GrassEntry {
    bool enabled = true;
    std::string positionsPath;
    std::uint32_t count = 1u << 13;
    glm::vec3 center = glm::vec3(0.0f);
    glm::vec2 extent = glm::vec2(15.0f);
    glm::vec3 position = glm::vec3(0.0f);
    glm::vec3 rotationDegrees = glm::vec3(0.0f);
    glm::vec3 scale = glm::vec3(1.0f);
    float minHeight = 1.3f;
    float maxHeight = 2.5f;
    float minWidth = 0.1f;
    float maxWidth = 0.14f;
    float minStiffness = 7.0f;
    float maxStiffness = 13.0f;
    glm::vec3 bottomColor = glm::vec3(0.1f, 0.4f, 0.1f);
    glm::vec3 topColor = glm::vec3(0.3f, 0.8f, 0.3f);
    glm::vec3 rimColor = glm::vec3(0.8f, 1.0f, 0.6f);
    bool flowersEnabled = false;
    float flowerDensity = 0.12f;
    float flowerHeightScale = 1.35f;
    std::uint32_t seed = 1337u;
};

struct CameraEntry {
    bool enabled = false;
    glm::vec3 position = glm::vec3(0.0f, 1.0f, 10.0f);
    glm::vec3 target = glm::vec3(0.0f, 1.0f, 0.0f);
    float fieldOfViewDegrees = 45.0f;
    float nearPlane = 0.1f;
    float farPlane = 100.0f;
};

struct Data {
    std::vector<ModelEntry> models;
    bool showGround = true;
    bool showGrass = true;
    EnvironmentEntry environment;
    GrassEntry grass;
    CameraEntry camera;
};

bool Load(const std::string& path, Data& output, std::string& error);

} // namespace SceneConfig
