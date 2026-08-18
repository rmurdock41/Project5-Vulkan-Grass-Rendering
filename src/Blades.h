#pragma once
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <array>
#include <cstdint>
#include <vector>
#include "Model.h"

struct BladeGenerationSettings {
    std::uint32_t count = 1u << 13;
    glm::vec3 center = glm::vec3(0.0f);
    glm::vec2 extent = glm::vec2(15.0f);
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

struct FlowerVertex {
    glm::vec3 position;
    glm::vec3 normal;
    float materialId;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription description = {};
        description.binding = 0;
        description.stride = sizeof(FlowerVertex);
        description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return description;
    }

    static std::array<VkVertexInputAttributeDescription, 3>
    getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 3> descriptions = {};
        descriptions[0].binding = 0;
        descriptions[0].location = 0;
        descriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        descriptions[0].offset = offsetof(FlowerVertex, position);
        descriptions[1].binding = 0;
        descriptions[1].location = 1;
        descriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        descriptions[1].offset = offsetof(FlowerVertex, normal);
        descriptions[2].binding = 0;
        descriptions[2].location = 2;
        descriptions[2].format = VK_FORMAT_R32_SFLOAT;
        descriptions[2].offset = offsetof(FlowerVertex, materialId);
        return descriptions;
    }
};

struct Blade {
    // Position and direction
    glm::vec4 v0;
    // Bezier point and height
    glm::vec4 v1;
    // Physical model guide and width
    glm::vec4 v2;
    // Up vector and stiffness coefficient
    glm::vec4 up;
    // Persistent tip velocity used by the semi-implicit physics integrator.
    // w is reserved to preserve std430 vec4 alignment.
    glm::vec4 velocity;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription = {};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Blade);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions = {};

        // v0
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Blade, v0);

        // v1
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Blade, v1);

        // v2
        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Blade, v2);

        // up
        attributeDescriptions[3].binding = 0;
        attributeDescriptions[3].location = 3;
        attributeDescriptions[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[3].offset = offsetof(Blade, up);

        return attributeDescriptions;
    }
};

struct BladeDrawIndirect {
    uint32_t vertexCount;
    uint32_t instanceCount;
    uint32_t firstVertex;
    uint32_t firstInstance;
};

class Blades : public Model {
private:
    VkBuffer bladesBuffer;
    VkBuffer culledBladesBuffer;
    VkBuffer numBladesBuffer;
    VkBuffer culledFlowersBuffer;
    VkBuffer flowerDrawIndirectBuffer;
    VkBuffer flowerVertexBuffer;
    VkBuffer flowerIndexBuffer;

    VkDeviceMemory bladesBufferMemory;
    VkDeviceMemory culledBladesBufferMemory;
    VkDeviceMemory numBladesBufferMemory;
    VkDeviceMemory culledFlowersBufferMemory;
    VkDeviceMemory flowerDrawIndirectBufferMemory;
    VkDeviceMemory flowerVertexBufferMemory;
    VkDeviceMemory flowerIndexBufferMemory;

    std::uint32_t bladeCount = 0;
    std::uint32_t flowerIndexCount = 0;
    bool flowersEnabled = false;
    float flowerDensity = 0.0f;
    float flowerHeightScale = 1.0f;

public:
    Blades(Device* device, VkCommandPool commandPool,
           const BladeGenerationSettings& settings,
           const std::vector<glm::vec3>& terrainPositions = {});
    VkBuffer GetBladesBuffer() const;
    VkBuffer GetCulledBladesBuffer() const;
    VkBuffer GetNumBladesBuffer() const;
    VkBuffer GetCulledFlowersBuffer() const;
    VkBuffer GetFlowerDrawIndirectBuffer() const;
    VkBuffer GetFlowerVertexBuffer() const;
    VkBuffer GetFlowerIndexBuffer() const;
    std::uint32_t GetBladeCount() const;
    std::uint32_t GetFlowerIndexCount() const;
    bool GetFlowersEnabled() const;
    float GetFlowerDensity() const;
    float GetFlowerHeightScale() const;
    ~Blades();
};
