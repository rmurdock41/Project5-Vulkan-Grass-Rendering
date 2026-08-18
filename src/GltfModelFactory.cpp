#include "GltfModelFactory.h"

#include "Image.h"

#include <glm/gtc/matrix_inverse.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>

namespace Gltf {

namespace {

float AlphaModeValue(const std::string& alphaMode) {
    if (alphaMode == "MASK") {
        return 1.0f;
    }
    if (alphaMode == "BLEND") {
        return 2.0f;
    }
    return 0.0f;
}

glm::vec3 SafeNormalize(const glm::vec3& value,
                        const glm::vec3& fallback) {
    const float lengthSquared = glm::dot(value, value);
    if (lengthSquared < 1.0e-12f) {
        return fallback;
    }
    return value / std::sqrt(lengthSquared);
}

glm::vec3 FallbackTangent(const glm::vec3& normal) {
    const glm::vec3 reference = std::abs(normal.y) < 0.999f
        ? glm::vec3(0.0f, 1.0f, 0.0f)
        : glm::vec3(1.0f, 0.0f, 0.0f);
    return SafeNormalize(glm::cross(reference, normal),
                         glm::vec3(1.0f, 0.0f, 0.0f));
}

void GenerateTangents(std::vector<Vertex>& vertices,
                      const std::vector<uint32_t>& indices) {
    std::vector<glm::vec3> tangents(vertices.size(), glm::vec3(0.0f));
    std::vector<glm::vec3> bitangents(vertices.size(), glm::vec3(0.0f));

    for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
        const uint32_t i0 = indices[i];
        const uint32_t i1 = indices[i + 1];
        const uint32_t i2 = indices[i + 2];
        const glm::vec3 edge1 = vertices[i1].pos - vertices[i0].pos;
        const glm::vec3 edge2 = vertices[i2].pos - vertices[i0].pos;
        const glm::vec2 uv1 = vertices[i1].texCoord - vertices[i0].texCoord;
        const glm::vec2 uv2 = vertices[i2].texCoord - vertices[i0].texCoord;
        const float determinant = uv1.x * uv2.y - uv1.y * uv2.x;
        if (std::abs(determinant) < 1.0e-10f) {
            continue;
        }
        const float inverseDeterminant = 1.0f / determinant;
        const glm::vec3 tangent =
            (edge1 * uv2.y - edge2 * uv1.y) * inverseDeterminant;
        const glm::vec3 bitangent =
            (edge2 * uv1.x - edge1 * uv2.x) * inverseDeterminant;
        tangents[i0] += tangent;
        tangents[i1] += tangent;
        tangents[i2] += tangent;
        bitangents[i0] += bitangent;
        bitangents[i1] += bitangent;
        bitangents[i2] += bitangent;
    }

    for (std::size_t i = 0; i < vertices.size(); ++i) {
        const glm::vec3 normal = SafeNormalize(
            vertices[i].normal, glm::vec3(0.0f, 1.0f, 0.0f));
        glm::vec3 tangent = tangents[i] -
            normal * glm::dot(normal, tangents[i]);
        tangent = SafeNormalize(tangent, FallbackTangent(normal));
        const float handedness =
            glm::dot(glm::cross(normal, tangent), bitangents[i]) < 0.0f
                ? -1.0f
                : 1.0f;
        vertices[i].tangent = glm::vec4(tangent, handedness);
    }
}

VkImage AcquireImage(Device* device, VkCommandPool commandPool,
                     const SceneData& scene, int textureIndex,
                     VkFormat format, VkImage fallback,
                     StaticModelResources& resources,
                     bool& usingFallback, uint32_t& mipLevels) {
    usingFallback = true;
    mipLevels = 1;
    if (textureIndex < 0 ||
        textureIndex >= static_cast<int>(scene.textures.size())) {
        return fallback;
    }
    const int imageIndex = scene.textures[textureIndex].imageIndex;
    if (imageIndex < 0 || imageIndex >= static_cast<int>(scene.images.size())) {
        return fallback;
    }

    usingFallback = false;
    for (std::size_t i = 0; i < resources.textureImages.size(); ++i) {
        if (resources.textureSourceIndices[i] == imageIndex &&
            resources.textureFormats[i] == format) {
            mipLevels = resources.textureMipLevels[i];
            return resources.textureImages[i];
        }
    }

    const ImageData& source = scene.images[imageIndex];
    if (source.encodedBytes.empty()) {
        throw std::runtime_error("Material image has no encoded data: " +
                                 source.name);
    }

    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    Image::FromEncodedMemory(device, commandPool, source.encodedBytes.data(),
                             source.encodedBytes.size(), image, memory, format,
                             &mipLevels);
    resources.textureImages.push_back(image);
    resources.textureMemories.push_back(memory);
    resources.textureSourceIndices.push_back(imageIndex);
    resources.textureFormats.push_back(format);
    resources.textureMipLevels.push_back(mipLevels);
    return image;
}

void BindMaterialTexture(Model* model, MaterialTextureSlot slot,
                         Device* device, VkCommandPool commandPool,
                         const SceneData& scene, int textureIndex,
                         VkFormat sourceFormat, VkImage fallback,
                         StaticModelResources& resources) {
    bool usingFallback = true;
    uint32_t mipLevels = 1;
    const VkImage image = AcquireImage(
        device, commandPool, scene, textureIndex, sourceFormat, fallback,
        resources, usingFallback, mipLevels);
    model->SetTexture(slot, image,
                      usingFallback ? VK_FORMAT_R8G8B8A8_UNORM
                                    : sourceFormat,
                      mipLevels);
}

struct PrimitiveBounds {
    glm::vec3 minimum = glm::vec3(0.0f);
    glm::vec3 maximum = glm::vec3(0.0f);
    bool valid = false;
};

PrimitiveBounds ComputePrimitiveBounds(const PrimitiveData& primitive,
                                       const glm::mat4& transform) {
    PrimitiveBounds bounds;
    for (std::size_t vertexIndex = 0;
         vertexIndex < primitive.vertices.size(); ++vertexIndex) {
        const glm::vec3 position = glm::vec3(
            transform * glm::vec4(
                primitive.vertices[vertexIndex].position, 1.0f));
        if (!bounds.valid) {
            bounds.minimum = position;
            bounds.maximum = position;
            bounds.valid = true;
        } else {
            bounds.minimum = glm::min(bounds.minimum, position);
            bounds.maximum = glm::max(bounds.maximum, position);
        }
    }
    return bounds;
}

const MaterialData* FindMaterial(const SceneData& scene,
                                 int materialIndex) {
    if (materialIndex < 0 ||
        materialIndex >= static_cast<int>(scene.materials.size())) {
        return nullptr;
    }
    return &scene.materials[materialIndex];
}

struct BatchKey {
    int materialIndex = -1;
    int cellX = 0;
    int cellY = 0;
    int cellZ = 0;

    bool operator<(const BatchKey& other) const {
        if (materialIndex != other.materialIndex) {
            return materialIndex < other.materialIndex;
        }
        if (cellX != other.cellX) {
            return cellX < other.cellX;
        }
        if (cellY != other.cellY) {
            return cellY < other.cellY;
        }
        return cellZ < other.cellZ;
    }
};

struct PendingBatch {
    int materialIndex = -1;
    std::vector<std::size_t> primitiveIndices;
    std::size_t vertexCount = 0;
    std::size_t indexCount = 0;
};

void AppendPrimitiveToBatch(PendingBatch& batch,
                            std::size_t primitiveIndex,
                            const PrimitiveData& primitive) {
    batch.primitiveIndices.push_back(primitiveIndex);
    batch.vertexCount += primitive.vertices.size();
    batch.indexCount += primitive.indices.size();
}

std::vector<PendingBatch> BuildPendingBatches(
        const SceneData& scene, const glm::mat4& sceneTransform,
        const StaticBatchingSettings& settings,
        StaticModelResources& resources) {
    std::vector<PendingBatch> batches;
    batches.reserve(scene.primitives.size());
    std::map<BatchKey, std::size_t> activeSpatialBatches;

    for (std::size_t primitiveIndex = 0;
         primitiveIndex < scene.primitives.size(); ++primitiveIndex) {
        const PrimitiveData& primitive = scene.primitives[primitiveIndex];
        if (primitive.vertices.empty() || primitive.indices.empty()) {
            continue;
        }
        ++resources.sourcePrimitiveCount;

        const MaterialData* material =
            FindMaterial(scene, primitive.materialIndex);
        const bool isBlend = material != nullptr &&
            material->alphaMode == "BLEND";
        const glm::mat4 transform =
            sceneTransform * primitive.nodeTransform;
        const PrimitiveBounds bounds =
            ComputePrimitiveBounds(primitive, transform);
        const glm::vec3 extent = bounds.valid
            ? bounds.maximum - bounds.minimum
            : glm::vec3(0.0f);
        const float maximumExtent =
            glm::max(extent.x, glm::max(extent.y, extent.z));
        const bool tooLarge =
            maximumExtent > settings.maximumPrimitiveExtent ||
            primitive.vertices.size() >
                settings.maximumVerticesPerBatch ||
            primitive.indices.size() > settings.maximumIndicesPerBatch;
        const bool canBatch = settings.enabled && !isBlend && !tooLarge;

        if (!canBatch) {
            PendingBatch batch;
            batch.materialIndex = primitive.materialIndex;
            AppendPrimitiveToBatch(batch, primitiveIndex, primitive);
            batches.push_back(batch);
            if (settings.enabled && isBlend) {
                ++resources.standaloneBlendPrimitiveCount;
            } else if (settings.enabled && tooLarge) {
                ++resources.standaloneLargePrimitiveCount;
            }
            continue;
        }

        const glm::vec3 center =
            (bounds.minimum + bounds.maximum) * 0.5f;
        const glm::vec3 cell = glm::floor(center / settings.cellSize);
        const BatchKey key = {
            primitive.materialIndex,
            static_cast<int>(cell.x),
            static_cast<int>(cell.y),
            static_cast<int>(cell.z)
        };

        std::map<BatchKey, std::size_t>::iterator active =
            activeSpatialBatches.find(key);
        bool createBatch = active == activeSpatialBatches.end();
        if (!createBatch) {
            const PendingBatch& batch = batches[active->second];
            createBatch =
                batch.vertexCount + primitive.vertices.size() >
                    settings.maximumVerticesPerBatch ||
                batch.indexCount + primitive.indices.size() >
                    settings.maximumIndicesPerBatch;
        }
        if (createBatch) {
            PendingBatch batch;
            batch.materialIndex = primitive.materialIndex;
            batches.push_back(batch);
            activeSpatialBatches[key] = batches.size() - 1;
            active = activeSpatialBatches.find(key);
        }
        AppendPrimitiveToBatch(
            batches[active->second], primitiveIndex, primitive);
    }

    for (std::size_t batchIndex = 0;
         batchIndex < batches.size(); ++batchIndex) {
        if (batches[batchIndex].primitiveIndices.size() > 1) {
            resources.mergedPrimitiveCount +=
                batches[batchIndex].primitiveIndices.size();
        }
    }
    resources.outputBatchCount = batches.size();
    return batches;
}

std::vector<Vertex> TransformPrimitiveVertices(
        const PrimitiveData& primitive, const glm::mat4& transform,
        const glm::vec4& baseColorFactor) {
    const glm::mat3 normalTransform =
        glm::inverseTranspose(glm::mat3(transform));
    const float transformSign =
        glm::determinant(glm::mat3(transform)) < 0.0f ? -1.0f : 1.0f;
    std::vector<Vertex> vertices;
    vertices.reserve(primitive.vertices.size());
    for (std::size_t vertexIndex = 0;
         vertexIndex < primitive.vertices.size(); ++vertexIndex) {
        const VertexData& source = primitive.vertices[vertexIndex];
        const glm::vec3 worldPosition = glm::vec3(
            transform * glm::vec4(source.position, 1.0f));
        const glm::vec3 worldNormal = primitive.hasNormals
            ? SafeNormalize(normalTransform * source.normal,
                            glm::vec3(0.0f, 1.0f, 0.0f))
            : glm::vec3(0.0f, 1.0f, 0.0f);
        glm::vec4 worldTangent(1.0f, 0.0f, 0.0f, 1.0f);
        if (primitive.hasTangents) {
            glm::vec3 tangentDirection = glm::mat3(transform) *
                glm::vec3(source.tangent);
            tangentDirection -= worldNormal *
                glm::dot(worldNormal, tangentDirection);
            tangentDirection = SafeNormalize(
                tangentDirection, FallbackTangent(worldNormal));
            worldTangent = glm::vec4(
                tangentDirection, source.tangent.w * transformSign);
        }
        vertices.push_back({worldPosition, glm::vec3(baseColorFactor),
                            source.texCoord, worldNormal, worldTangent});
    }
    if (!primitive.hasTangents) {
        GenerateTangents(vertices, primitive.indices);
    }
    return vertices;
}

} // namespace

void DestroyStaticModels(Device* device, StaticModelResources& resources) {
    for (std::size_t i = 0; i < resources.models.size(); ++i) {
        delete resources.models[i];
    }
    resources.models.clear();

    for (std::size_t i = 0; i < resources.textureImages.size(); ++i) {
        if (resources.textureImages[i] != VK_NULL_HANDLE) {
            vkDestroyImage(device->GetVkDevice(), resources.textureImages[i],
                           nullptr);
        }
        if (resources.textureMemories[i] != VK_NULL_HANDLE) {
            vkFreeMemory(device->GetVkDevice(), resources.textureMemories[i],
                         nullptr);
        }
    }
    resources.textureImages.clear();
    resources.textureMemories.clear();
    resources.textureSourceIndices.clear();
    resources.textureFormats.clear();
    resources.textureMipLevels.clear();
}

StaticModelResources CreateStaticModels(Device* device,
                                        VkCommandPool commandPool,
                                        const SceneData& scene,
                                        const TextureFallbacks& fallbacks,
                                        const glm::mat4& sceneTransform,
                                        const StaticBatchingSettings&
                                            batchingSettings) {
    StaticModelResources resources;

    try {
        const std::vector<PendingBatch> batches = BuildPendingBatches(
            scene, sceneTransform, batchingSettings, resources);
        resources.models.reserve(batches.size());

        for (std::size_t batchIndex = 0;
             batchIndex < batches.size(); ++batchIndex) {
            const PendingBatch& batch = batches[batchIndex];
            const MaterialData* material =
                FindMaterial(scene, batch.materialIndex);
            const glm::vec4 baseColorFactor = material != nullptr
                ? material->baseColorFactor
                : glm::vec4(1.0f);

            std::vector<Vertex> vertices;
            std::vector<uint32_t> indices;
            vertices.reserve(batch.vertexCount);
            indices.reserve(batch.indexCount);
            for (std::size_t itemIndex = 0;
                 itemIndex < batch.primitiveIndices.size(); ++itemIndex) {
                const PrimitiveData& primitive =
                    scene.primitives[batch.primitiveIndices[itemIndex]];
                const glm::mat4 transform =
                    sceneTransform * primitive.nodeTransform;
                std::vector<Vertex> primitiveVertices =
                    TransformPrimitiveVertices(
                        primitive, transform, baseColorFactor);
                const uint32_t vertexOffset =
                    static_cast<uint32_t>(vertices.size());
                vertices.insert(vertices.end(), primitiveVertices.begin(),
                                primitiveVertices.end());
                for (std::size_t index = 0;
                     index < primitive.indices.size(); ++index) {
                    indices.push_back(
                        primitive.indices[index] + vertexOffset);
                }
            }

            const float alphaCutoff = material != nullptr
                ? material->alphaCutoff
                : 0.5f;
            const float alphaMode = material != nullptr
                ? AlphaModeValue(material->alphaMode)
                : 0.0f;
            const glm::vec4 materialParameters(
                alphaCutoff, alphaMode, baseColorFactor.a, 0.0f);
            const glm::vec4 pbrParameters(
                material != nullptr ? material->metallicFactor : 0.0f,
                material != nullptr ? material->roughnessFactor : 1.0f,
                material != nullptr ? material->normalScale : 1.0f,
                material != nullptr ? material->specularFactor : 1.0f);
            const glm::vec3 specularColor = material != nullptr
                ? material->specularColorFactor
                : glm::vec3(1.0f);
            const glm::vec4 specularColorParameters(
                specularColor,
                material != nullptr && material->doubleSided ? 1.0f : 0.0f);
            std::unique_ptr<Model> model(new Model(
                device, commandPool, vertices, indices,
                materialParameters, pbrParameters,
                specularColorParameters));

            BindMaterialTexture(
                model.get(), MaterialTextureSlot::BaseColor, device,
                commandPool, scene,
                material != nullptr ? material->baseColorTexture : -1,
                VK_FORMAT_R8G8B8A8_SRGB, fallbacks.baseColor, resources);
            BindMaterialTexture(
                model.get(), MaterialTextureSlot::Normal, device, commandPool,
                scene, material != nullptr ? material->normalTexture : -1,
                VK_FORMAT_R8G8B8A8_UNORM, fallbacks.normal, resources);
            BindMaterialTexture(
                model.get(), MaterialTextureSlot::MetallicRoughness, device,
                commandPool, scene,
                material != nullptr ? material->metallicRoughnessTexture : -1,
                VK_FORMAT_R8G8B8A8_UNORM, fallbacks.metallicRoughness,
                resources);
            BindMaterialTexture(
                model.get(), MaterialTextureSlot::Specular, device,
                commandPool, scene,
                material != nullptr ? material->specularTexture : -1,
                VK_FORMAT_R8G8B8A8_UNORM, fallbacks.specular, resources);
            BindMaterialTexture(
                model.get(), MaterialTextureSlot::SpecularColor, device,
                commandPool, scene,
                material != nullptr ? material->specularColorTexture : -1,
                VK_FORMAT_R8G8B8A8_SRGB, fallbacks.specularColor, resources);

            resources.models.push_back(model.release());
        }
    } catch (...) {
        DestroyStaticModels(device, resources);
        throw;
    }

    return resources;
}

} // namespace Gltf
