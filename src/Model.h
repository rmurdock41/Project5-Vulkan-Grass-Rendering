#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <array>
#include <vector>

#include "Vertex.h"
#include "Device.h"

struct ModelBufferObject {
    glm::mat4 modelMatrix;
    // x: alpha cutoff, y: 0=opaque/1=mask/2=blend, z: alpha factor.
    glm::vec4 materialParameters;
    // x: metallic, y: roughness, z: normal scale, w: specular factor.
    glm::vec4 pbrParameters;
    // rgb: dielectric specular tint, w: double-sided material flag.
    glm::vec4 specularColorParameters;
};

enum class MaterialTextureSlot : uint32_t {
    BaseColor = 0,
    Normal = 1,
    MetallicRoughness = 2,
    Specular = 3,
    SpecularColor = 4,
    Count = 5
};

enum class MaterialAlphaMode : uint32_t {
    Opaque = 0,
    Mask = 1,
    Blend = 2
};

class Model {
protected:
    Device* device;

    std::vector<Vertex> vertices;
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;

    std::vector<uint32_t> indices;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;

    VkBuffer modelBuffer;
    VkDeviceMemory modelBufferMemory;

    ModelBufferObject modelBufferObject;

    glm::vec3 boundsMin;
    glm::vec3 boundsMax;
    bool hasBounds;

    static const std::size_t MaterialTextureCount =
        static_cast<std::size_t>(MaterialTextureSlot::Count);
    std::array<VkImage, MaterialTextureCount> textures;
    std::array<VkImageView, MaterialTextureCount> textureViews;
    std::array<VkSampler, MaterialTextureCount> textureSamplers;
    std::array<uint32_t, MaterialTextureCount> textureMipLevels;

public:
    Model() = delete;
    Model(Device* device, VkCommandPool commandPool, const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices, const glm::vec4& materialParameters = glm::vec4(0.5f, 0.0f, 1.0f, 0.0f), const glm::vec4& pbrParameters = glm::vec4(0.0f, 1.0f, 1.0f, 1.0f), const glm::vec4& specularColorParameters = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f));
    virtual ~Model();

    void SetTexture(VkImage texture, VkFormat format = VK_FORMAT_R8G8B8A8_UNORM,
                    uint32_t mipLevels = 1);
    void SetTexture(MaterialTextureSlot slot, VkImage texture,
                    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM,
                    uint32_t mipLevels = 1);

    const std::vector<Vertex>& getVertices() const;

    VkBuffer getVertexBuffer() const;

    const std::vector<uint32_t>& getIndices() const;

    VkBuffer getIndexBuffer() const;

    const ModelBufferObject& getModelBufferObject() const;

    MaterialAlphaMode GetAlphaMode() const;
    bool IsDoubleSided() const;

    bool HasBounds() const;
    const glm::vec3& GetBoundsMin() const;
    const glm::vec3& GetBoundsMax() const;

    VkBuffer GetModelBuffer() const;
    VkImageView GetTextureView(MaterialTextureSlot slot = MaterialTextureSlot::BaseColor) const;
    VkSampler GetTextureSampler(MaterialTextureSlot slot = MaterialTextureSlot::BaseColor) const;
};
