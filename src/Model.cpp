#include "Model.h"
#include "BufferUtils.h"
#include "Image.h"

#include <algorithm>
#include <limits>

Model::Model(Device* device, VkCommandPool commandPool, const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices, const glm::vec4& materialParameters, const glm::vec4& pbrParameters, const glm::vec4& specularColorParameters)
  : device(device), vertices(vertices), indices(indices),
    boundsMin(0.0f), boundsMax(0.0f), hasBounds(!vertices.empty()) {

    textures.fill(VK_NULL_HANDLE);
    textureViews.fill(VK_NULL_HANDLE);
    textureSamplers.fill(VK_NULL_HANDLE);
    textureMipLevels.fill(1);

    if (hasBounds) {
        const float maximum = std::numeric_limits<float>::max();
        boundsMin = glm::vec3(maximum);
        boundsMax = glm::vec3(-maximum);
        for (std::size_t index = 0; index < vertices.size(); ++index) {
            boundsMin = glm::min(boundsMin, vertices[index].pos);
            boundsMax = glm::max(boundsMax, vertices[index].pos);
        }
    }

    if (vertices.size() > 0) {
        BufferUtils::CreateBufferFromData(device, commandPool, this->vertices.data(), vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBuffer, vertexBufferMemory);
    }

    if (indices.size() > 0) {
        BufferUtils::CreateBufferFromData(device, commandPool, this->indices.data(), indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBuffer, indexBufferMemory);
    }

    modelBufferObject.modelMatrix = glm::mat4(1.0f);
    modelBufferObject.materialParameters = materialParameters;
    modelBufferObject.pbrParameters = pbrParameters;
    modelBufferObject.specularColorParameters = specularColorParameters;
    BufferUtils::CreateBufferFromData(device, commandPool, &modelBufferObject, sizeof(ModelBufferObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, modelBuffer, modelBufferMemory);
}

Model::~Model() {
    if (indices.size() > 0) {
        vkDestroyBuffer(device->GetVkDevice(), indexBuffer, nullptr);
        vkFreeMemory(device->GetVkDevice(), indexBufferMemory, nullptr);
    }

    if (vertices.size() > 0) {
        vkDestroyBuffer(device->GetVkDevice(), vertexBuffer, nullptr);
        vkFreeMemory(device->GetVkDevice(), vertexBufferMemory, nullptr);
    }

    vkDestroyBuffer(device->GetVkDevice(), modelBuffer, nullptr);
    vkFreeMemory(device->GetVkDevice(), modelBufferMemory, nullptr);

    for (std::size_t i = 0; i < MaterialTextureCount; ++i) {
        if (textureViews[i] != VK_NULL_HANDLE) {
            vkDestroyImageView(device->GetVkDevice(), textureViews[i], nullptr);
        }
        if (textureSamplers[i] != VK_NULL_HANDLE) {
            vkDestroySampler(device->GetVkDevice(), textureSamplers[i], nullptr);
        }
    }
}

void Model::SetTexture(VkImage texture, VkFormat format,
                       uint32_t mipLevels) {
    SetTexture(MaterialTextureSlot::BaseColor, texture, format, mipLevels);
}

void Model::SetTexture(MaterialTextureSlot slot, VkImage texture,
                       VkFormat format, uint32_t mipLevels) {
    const std::size_t slotIndex = static_cast<std::size_t>(slot);
    mipLevels = std::max(1u, mipLevels);
    textures[slotIndex] = texture;
    textureMipLevels[slotIndex] = mipLevels;
    textureViews[slotIndex] = Image::CreateView(
        device, texture, format, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);

    // --- Specify all filters and transformations ---
    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;

    // Interpolation of texels that are magnified or minified
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;

    // Addressing mode
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

    // Anisotropic filtering
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = 16;

    // Border color
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;

    // Choose coordinate system for addressing texels --> [0, 1) here
    samplerInfo.unnormalizedCoordinates = VK_FALSE;

    // Comparison function used for filtering operations
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;

    // Mipmapping
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = static_cast<float>(mipLevels - 1);

    if (vkCreateSampler(device->GetVkDevice(), &samplerInfo, nullptr, &textureSamplers[slotIndex]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create texture sampler");
    }
}

const std::vector<Vertex>& Model::getVertices() const {
    return vertices;
}

VkBuffer Model::getVertexBuffer() const {
    return vertexBuffer;
}

const std::vector<uint32_t>& Model::getIndices() const {
    return indices;
}

VkBuffer Model::getIndexBuffer() const {
    return indexBuffer;
}

const ModelBufferObject& Model::getModelBufferObject() const {
    return modelBufferObject;
}

MaterialAlphaMode Model::GetAlphaMode() const {
    const float alphaMode = modelBufferObject.materialParameters.y;
    if (alphaMode > 1.5f) {
        return MaterialAlphaMode::Blend;
    }
    if (alphaMode > 0.5f) {
        return MaterialAlphaMode::Mask;
    }
    return MaterialAlphaMode::Opaque;
}

bool Model::IsDoubleSided() const {
    return modelBufferObject.specularColorParameters.w > 0.5f;
}

bool Model::HasBounds() const {
    return hasBounds;
}

const glm::vec3& Model::GetBoundsMin() const {
    return boundsMin;
}

const glm::vec3& Model::GetBoundsMax() const {
    return boundsMax;
}

VkBuffer Model::GetModelBuffer() const {
    return modelBuffer;
}

VkImageView Model::GetTextureView(MaterialTextureSlot slot) const {
    return textureViews[static_cast<std::size_t>(slot)];
}

VkSampler Model::GetTextureSampler(MaterialTextureSlot slot) const {
    return textureSamplers[static_cast<std::size_t>(slot)];
}
