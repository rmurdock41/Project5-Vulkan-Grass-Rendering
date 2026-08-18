#pragma once

#include "GltfLoader.h"
#include "Model.h"

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include <vector>

namespace Gltf {

struct StaticModelResources {
    std::vector<Model*> models;
    // Images are shared when both glTF image index and color space match.
    std::vector<VkImage> textureImages;
    std::vector<VkDeviceMemory> textureMemories;
    std::vector<int> textureSourceIndices;
    std::vector<VkFormat> textureFormats;
    std::vector<uint32_t> textureMipLevels;
    std::size_t sourcePrimitiveCount = 0;
    std::size_t outputBatchCount = 0;
    std::size_t mergedPrimitiveCount = 0;
    std::size_t standaloneLargePrimitiveCount = 0;
    std::size_t standaloneBlendPrimitiveCount = 0;
};

struct StaticBatchingSettings {
    bool enabled = true;
    // World-space partition size. Primitives only merge inside the same cell.
    float cellSize = 4.0f;
    // Large primitives keep their own culling bounds instead of entering a
    // spatial batch.
    float maximumPrimitiveExtent = 6.0f;
    std::size_t maximumVerticesPerBatch = 500000;
    std::size_t maximumIndicesPerBatch = 1500000;
};

struct TextureFallbacks {
    VkImage baseColor = VK_NULL_HANDLE;
    VkImage normal = VK_NULL_HANDLE;
    VkImage metallicRoughness = VK_NULL_HANDLE;
    VkImage specular = VK_NULL_HANDLE;
    VkImage specularColor = VK_NULL_HANDLE;
};

StaticModelResources CreateStaticModels(Device* device,
                                        VkCommandPool commandPool,
                                        const SceneData& scene,
                                        const TextureFallbacks& fallbacks,
                                        const glm::mat4& sceneTransform,
                                        const StaticBatchingSettings&
                                            batchingSettings =
                                                StaticBatchingSettings());

void DestroyStaticModels(Device* device, StaticModelResources& resources);

} // namespace Gltf
