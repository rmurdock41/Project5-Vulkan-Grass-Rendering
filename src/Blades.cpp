#include <vector>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <cmath>
#include "Blades.h"
#include "BufferUtils.h"

Blades::Blades(Device* device, VkCommandPool commandPool,
               const BladeGenerationSettings& settings,
               const std::vector<glm::vec3>& terrainPositions)
    : Model(device, commandPool, {}, {},
            glm::vec4(settings.bottomColor, 1.0f),
            glm::vec4(settings.topColor, 1.0f),
            glm::vec4(settings.rimColor, 1.0f)) {
    flowersEnabled = settings.flowersEnabled;
    flowerDensity = settings.flowerDensity;
    flowerHeightScale = settings.flowerHeightScale;
    bladeCount = terrainPositions.empty()
        ? settings.count
        : std::min<std::uint32_t>(
              settings.count,
              static_cast<std::uint32_t>(terrainPositions.size()));
    if (bladeCount == 0) {
        throw std::runtime_error("Grass blade count must be greater than zero");
    }

    std::mt19937 random(settings.seed);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    std::vector<Blade> blades;
    blades.reserve(bladeCount);

    for (std::uint32_t i = 0; i < bladeCount; ++i) {
        Blade currentBlade = Blade();

        glm::vec3 bladeUp(0.0f, 1.0f, 0.0f);

        // Generate positions and direction (v0)
        glm::vec3 bladePosition;
        if (terrainPositions.empty()) {
            bladePosition = settings.center + glm::vec3(
                (unit(random) - 0.5f) * settings.extent.x,
                0.0f,
                (unit(random) - 0.5f) * settings.extent.y);
        } else {
            bladePosition = terrainPositions[i];
        }
        float direction = unit(random) * 2.f * 3.14159265f;
        currentBlade.v0 = glm::vec4(bladePosition, direction);

        // Bezier point and height (v1)
        float height = settings.minHeight +
            unit(random) * (settings.maxHeight - settings.minHeight);
        currentBlade.v1 = glm::vec4(bladePosition + bladeUp * height, height); 

        // Physical model guide and width (v2)
        float width = settings.minWidth +
            unit(random) * (settings.maxWidth - settings.minWidth);
        float bendAmount = height * 0.12f;
        glm::vec3 bendOffset = glm::vec3(
            (unit(random) - 0.5f) * bendAmount,
            0.0f,
            (unit(random) - 0.5f) * bendAmount
        );
        currentBlade.v2 = glm::vec4(bladePosition + bladeUp * height + bendOffset, width);  

        // Up vector and stiffness coefficient (up)
        float stiffness = settings.minStiffness +
            unit(random) * (settings.maxStiffness - settings.minStiffness);
        currentBlade.up = glm::vec4(bladeUp, stiffness);
        currentBlade.velocity = glm::vec4(0.0f);

        blades.push_back(currentBlade);
    }

    BladeDrawIndirect indirectDraw;
    indirectDraw.vertexCount = bladeCount;
    indirectDraw.instanceCount = 1;
    indirectDraw.firstVertex = 0;
    indirectDraw.firstInstance = 0;

    BufferUtils::CreateBufferFromData(device, commandPool, blades.data(), bladeCount * sizeof(Blade),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        bladesBuffer, bladesBufferMemory);


    BufferUtils::CreateBuffer(device, bladeCount * sizeof(Blade),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, culledBladesBuffer, culledBladesBufferMemory);

    BufferUtils::CreateBufferFromData(device, commandPool, &indirectDraw, sizeof(BladeDrawIndirect), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, numBladesBuffer, numBladesBufferMemory);

    // The compute pass compacts visible flower instances into this buffer.
    // It deliberately stores the same Blade payload as grass, so flowers
    // inherit the existing wind state and culling result without a second
    // simulation pass.
    BufferUtils::CreateBuffer(
        device, bladeCount * sizeof(Blade),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        culledFlowersBuffer, culledFlowersBufferMemory);

    std::vector<FlowerVertex> flowerVertices;
    std::vector<std::uint32_t> flowerIndices;
    const float pi = 3.14159265358979323846f;

    // Six-sided tapered stem.  Its local Y range is normalized to [0, 1];
    // flower.vert bends that axis along the simulated grass Bezier curve.
    const std::uint32_t stemSides = 6;
    for (std::uint32_t side = 0; side < stemSides; ++side) {
        const float angle = 2.0f * pi * static_cast<float>(side) /
            static_cast<float>(stemSides);
        const glm::vec3 normal(std::cos(angle), 0.0f, std::sin(angle));
        flowerVertices.push_back({ glm::vec3(normal.x * 0.025f, 0.0f,
                                               normal.z * 0.025f),
                                   normal, 0.0f });
        flowerVertices.push_back({ glm::vec3(normal.x * 0.015f, 1.0f,
                                               normal.z * 0.015f),
                                   normal, 0.0f });
    }
    for (std::uint32_t side = 0; side < stemSides; ++side) {
        const std::uint32_t next = (side + 1) % stemSides;
        flowerIndices.push_back(side * 2);
        flowerIndices.push_back(next * 2);
        flowerIndices.push_back(side * 2 + 1);
        flowerIndices.push_back(side * 2 + 1);
        flowerIndices.push_back(next * 2);
        flowerIndices.push_back(next * 2 + 1);
    }

    // Five broad, slightly irregular star petals.  This is intentionally a
    // tiny shared mesh rather than five textured alpha cards, so it stays
    // stable at distance and reads like the pale Lumenflower field.
    const std::uint32_t petalCount = 5;
    for (std::uint32_t petal = 0; petal < petalCount; ++petal) {
        const float angle = 2.0f * pi * static_cast<float>(petal) /
            static_cast<float>(petalCount);
        const glm::vec3 radial(std::cos(angle), 0.0f, std::sin(angle));
        const glm::vec3 lateral(-radial.z, 0.0f, radial.x);
        const std::uint32_t base =
            static_cast<std::uint32_t>(flowerVertices.size());
        flowerVertices.push_back({ radial * 0.025f + glm::vec3(0.0f, 1.0f, 0.0f),
                                   glm::vec3(0.0f, 1.0f, 0.0f), 1.0f });
        flowerVertices.push_back({ radial * 0.13f - lateral * 0.085f +
                                       glm::vec3(0.0f, 1.0f, 0.0f),
                                   glm::vec3(0.0f, 1.0f, 0.0f), 1.0f });
        flowerVertices.push_back({ radial * 0.29f +
                                       glm::vec3(0.0f, 1.0f, 0.0f),
                                   glm::vec3(0.0f, 1.0f, 0.0f), 1.0f });
        flowerVertices.push_back({ radial * 0.13f + lateral * 0.085f +
                                       glm::vec3(0.0f, 1.0f, 0.0f),
                                   glm::vec3(0.0f, 1.0f, 0.0f), 1.0f });
        flowerIndices.insert(flowerIndices.end(), {
            base, base + 1, base + 2,
            base, base + 2, base + 3
        });
    }

    // Small warm centre disc, contrasting the cool white petals.
    const std::uint32_t center =
        static_cast<std::uint32_t>(flowerVertices.size());
    flowerVertices.push_back({ glm::vec3(0.0f, 1.003f, 0.0f),
                               glm::vec3(0.0f, 1.0f, 0.0f), 2.0f });
    const std::uint32_t centerSides = 8;
    for (std::uint32_t side = 0; side < centerSides; ++side) {
        const float angle = 2.0f * pi * static_cast<float>(side) /
            static_cast<float>(centerSides);
        flowerVertices.push_back({
            glm::vec3(std::cos(angle) * 0.052f, 1.004f,
                      std::sin(angle) * 0.052f),
            glm::vec3(0.0f, 1.0f, 0.0f), 2.0f });
    }
    for (std::uint32_t side = 0; side < centerSides; ++side) {
        flowerIndices.push_back(center);
        flowerIndices.push_back(center + 1 + side);
        flowerIndices.push_back(center + 1 + (side + 1) % centerSides);
    }

    flowerIndexCount = static_cast<std::uint32_t>(flowerIndices.size());
    BufferUtils::CreateBufferFromData(
        device, commandPool, flowerVertices.data(),
        flowerVertices.size() * sizeof(FlowerVertex),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        flowerVertexBuffer, flowerVertexBufferMemory);
    BufferUtils::CreateBufferFromData(
        device, commandPool, flowerIndices.data(),
        flowerIndices.size() * sizeof(std::uint32_t),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        flowerIndexBuffer, flowerIndexBufferMemory);

    VkDrawIndexedIndirectCommand flowerDraw = {};
    flowerDraw.indexCount = flowerIndexCount;
    flowerDraw.instanceCount = 0;
    BufferUtils::CreateBufferFromData(
        device, commandPool, &flowerDraw, sizeof(flowerDraw),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        flowerDrawIndirectBuffer, flowerDrawIndirectBufferMemory);
}

VkBuffer Blades::GetBladesBuffer() const {
    return bladesBuffer;
}

VkBuffer Blades::GetCulledBladesBuffer() const {
    return culledBladesBuffer;
}

VkBuffer Blades::GetNumBladesBuffer() const {
    return numBladesBuffer;
}

VkBuffer Blades::GetCulledFlowersBuffer() const {
    return culledFlowersBuffer;
}

VkBuffer Blades::GetFlowerDrawIndirectBuffer() const {
    return flowerDrawIndirectBuffer;
}

VkBuffer Blades::GetFlowerVertexBuffer() const {
    return flowerVertexBuffer;
}

VkBuffer Blades::GetFlowerIndexBuffer() const {
    return flowerIndexBuffer;
}

std::uint32_t Blades::GetBladeCount() const {
    return bladeCount;
}

std::uint32_t Blades::GetFlowerIndexCount() const {
    return flowerIndexCount;
}

bool Blades::GetFlowersEnabled() const {
    return flowersEnabled;
}

float Blades::GetFlowerDensity() const {
    return flowerDensity;
}

float Blades::GetFlowerHeightScale() const {
    return flowerHeightScale;
}

Blades::~Blades() {
    vkDestroyBuffer(device->GetVkDevice(), bladesBuffer, nullptr);
    vkFreeMemory(device->GetVkDevice(), bladesBufferMemory, nullptr);
    vkDestroyBuffer(device->GetVkDevice(), culledBladesBuffer, nullptr);
    vkFreeMemory(device->GetVkDevice(), culledBladesBufferMemory, nullptr);
    vkDestroyBuffer(device->GetVkDevice(), numBladesBuffer, nullptr);
    vkFreeMemory(device->GetVkDevice(), numBladesBufferMemory, nullptr);
    vkDestroyBuffer(device->GetVkDevice(), culledFlowersBuffer, nullptr);
    vkFreeMemory(device->GetVkDevice(), culledFlowersBufferMemory, nullptr);
    vkDestroyBuffer(device->GetVkDevice(), flowerDrawIndirectBuffer, nullptr);
    vkFreeMemory(device->GetVkDevice(), flowerDrawIndirectBufferMemory, nullptr);
    vkDestroyBuffer(device->GetVkDevice(), flowerVertexBuffer, nullptr);
    vkFreeMemory(device->GetVkDevice(), flowerVertexBufferMemory, nullptr);
    vkDestroyBuffer(device->GetVkDevice(), flowerIndexBuffer, nullptr);
    vkFreeMemory(device->GetVkDevice(), flowerIndexBufferMemory, nullptr);
}
