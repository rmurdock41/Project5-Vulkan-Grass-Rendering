#include "EnvironmentMap.h"

#include "BufferUtils.h"
#include "Instance.h"

#include <stb_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace EnvironmentMap {

namespace {

const float Pi = 3.14159265358979323846f;

struct CubePixels {
    uint32_t faceSize = 1;
    uint32_t mipLevels = 1;
    std::vector<std::vector<float>> levels;
};

glm::vec3 FaceDirection(uint32_t face, float u, float v) {
    switch (face) {
    case 0: return glm::normalize(glm::vec3(1.0f, -v, -u));
    case 1: return glm::normalize(glm::vec3(-1.0f, -v, u));
    case 2: return glm::normalize(glm::vec3(u, 1.0f, v));
    case 3: return glm::normalize(glm::vec3(u, -1.0f, -v));
    case 4: return glm::normalize(glm::vec3(u, -v, 1.0f));
    default: return glm::normalize(glm::vec3(-u, -v, -1.0f));
    }
}

glm::vec4 SampleEquirectangular(const float* pixels, int width, int height,
                                const glm::vec3& direction) {
    float longitude = std::atan2(direction.z, direction.x);
    float latitude = std::asin(glm::clamp(direction.y, -1.0f, 1.0f));
    float x = (longitude / (2.0f * Pi) + 0.5f) * width - 0.5f;
    float y = (0.5f - latitude / Pi) * height - 0.5f;

    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    const float tx = x - std::floor(x);
    const float ty = y - std::floor(y);
    x0 = (x0 % width + width) % width;
    y0 = std::max(0, std::min(height - 1, y0));
    const int x1 = (x0 + 1) % width;
    const int y1 = std::min(height - 1, y0 + 1);

    const glm::vec4 c00 = glm::make_vec4(
        pixels + (static_cast<std::size_t>(y0) * width + x0) * 4);
    const glm::vec4 c10 = glm::make_vec4(
        pixels + (static_cast<std::size_t>(y0) * width + x1) * 4);
    const glm::vec4 c01 = glm::make_vec4(
        pixels + (static_cast<std::size_t>(y1) * width + x0) * 4);
    const glm::vec4 c11 = glm::make_vec4(
        pixels + (static_cast<std::size_t>(y1) * width + x1) * 4);
    return glm::mix(glm::mix(c00, c10, tx), glm::mix(c01, c11, tx), ty);
}

uint32_t CalculateMipLevels(uint32_t size) {
    uint32_t levels = 1;
    while (size > 1) {
        size /= 2;
        ++levels;
    }
    return levels;
}

CubePixels BuildCubePixels(const std::string& path,
                           uint32_t maximumFaceSize) {
    CubePixels cube;
    if (path.empty()) {
        cube.levels.resize(6);
        for (uint32_t face = 0; face < 6; ++face) {
            cube.levels[face] = { 0.25f, 0.45f, 0.8f, 1.0f };
        }
        return cube;
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    float* source = stbi_loadf(path.c_str(), &width, &height, &channels, 4);
    if (source == nullptr || width <= 0 || height <= 0) {
        throw std::runtime_error(
            "Failed to load HDR environment '" + path + "': " +
            (stbi_failure_reason() != nullptr ? stbi_failure_reason()
                                              : "unknown stb_image error"));
    }

    cube.faceSize = std::max(
        1u, std::min(maximumFaceSize, static_cast<uint32_t>(height / 2)));
    cube.mipLevels = CalculateMipLevels(cube.faceSize);
    cube.levels.resize(static_cast<std::size_t>(cube.mipLevels) * 6);

    try {
        for (uint32_t face = 0; face < 6; ++face) {
            std::vector<float>& base = cube.levels[face];
            base.resize(static_cast<std::size_t>(cube.faceSize) *
                        cube.faceSize * 4);
            for (uint32_t y = 0; y < cube.faceSize; ++y) {
                for (uint32_t x = 0; x < cube.faceSize; ++x) {
                    const float u =
                        2.0f * (static_cast<float>(x) + 0.5f) /
                            cube.faceSize -
                        1.0f;
                    const float v =
                        2.0f * (static_cast<float>(y) + 0.5f) /
                            cube.faceSize -
                        1.0f;
                    const glm::vec4 color = SampleEquirectangular(
                        source, width, height, FaceDirection(face, u, v));
                    std::memcpy(
                        &base[(static_cast<std::size_t>(y) * cube.faceSize +
                               x) *
                              4],
                        &color[0], sizeof(float) * 4);
                }
            }
        }
    } catch (...) {
        stbi_image_free(source);
        throw;
    }
    stbi_image_free(source);

    uint32_t sourceSize = cube.faceSize;
    for (uint32_t mip = 1; mip < cube.mipLevels; ++mip) {
        const uint32_t destinationSize = std::max(1u, sourceSize / 2);
        for (uint32_t face = 0; face < 6; ++face) {
            const std::vector<float>& previous =
                cube.levels[(mip - 1) * 6 + face];
            std::vector<float>& current = cube.levels[mip * 6 + face];
            current.resize(static_cast<std::size_t>(destinationSize) *
                           destinationSize * 4);
            for (uint32_t y = 0; y < destinationSize; ++y) {
                for (uint32_t x = 0; x < destinationSize; ++x) {
                    glm::vec4 sum(0.0f);
                    for (uint32_t oy = 0; oy < 2; ++oy) {
                        for (uint32_t ox = 0; ox < 2; ++ox) {
                            const uint32_t sx =
                                std::min(sourceSize - 1, x * 2 + ox);
                            const uint32_t sy =
                                std::min(sourceSize - 1, y * 2 + oy);
                            sum += glm::make_vec4(
                                &previous[(static_cast<std::size_t>(sy) *
                                               sourceSize +
                                           sx) *
                                          4]);
                        }
                    }
                    const glm::vec4 color = sum * 0.25f;
                    std::memcpy(
                        &current[(static_cast<std::size_t>(y) *
                                      destinationSize +
                                  x) *
                                 4],
                        &color[0], sizeof(float) * 4);
                }
            }
        }
        sourceSize = destinationSize;
    }
    return cube;
}

glm::vec3 SampleCube(const CubePixels& cube, const glm::vec3& rawDirection) {
    const glm::vec3 direction = glm::normalize(rawDirection);
    const glm::vec3 absolute = glm::abs(direction);
    uint32_t face = 0;
    float u = 0.0f;
    float v = 0.0f;
    if (absolute.x >= absolute.y && absolute.x >= absolute.z) {
        if (direction.x >= 0.0f) {
            face = 0;
            u = -direction.z / absolute.x;
            v = -direction.y / absolute.x;
        } else {
            face = 1;
            u = direction.z / absolute.x;
            v = -direction.y / absolute.x;
        }
    } else if (absolute.y >= absolute.z) {
        if (direction.y >= 0.0f) {
            face = 2;
            u = direction.x / absolute.y;
            v = direction.z / absolute.y;
        } else {
            face = 3;
            u = direction.x / absolute.y;
            v = -direction.z / absolute.y;
        }
    } else if (direction.z >= 0.0f) {
        face = 4;
        u = direction.x / absolute.z;
        v = -direction.y / absolute.z;
    } else {
        face = 5;
        u = -direction.x / absolute.z;
        v = -direction.y / absolute.z;
    }

    const float pixelX =
        (u * 0.5f + 0.5f) * cube.faceSize - 0.5f;
    const float pixelY =
        (v * 0.5f + 0.5f) * cube.faceSize - 0.5f;
    const int x0 = std::max(
        0, std::min(static_cast<int>(cube.faceSize) - 1,
                    static_cast<int>(std::floor(pixelX))));
    const int y0 = std::max(
        0, std::min(static_cast<int>(cube.faceSize) - 1,
                    static_cast<int>(std::floor(pixelY))));
    const int x1 = std::min(static_cast<int>(cube.faceSize) - 1, x0 + 1);
    const int y1 = std::min(static_cast<int>(cube.faceSize) - 1, y0 + 1);
    const float tx = glm::clamp(pixelX - std::floor(pixelX), 0.0f, 1.0f);
    const float ty = glm::clamp(pixelY - std::floor(pixelY), 0.0f, 1.0f);
    const std::vector<float>& pixels = cube.levels[face];
    const auto sample = [&](int x, int y) {
        return glm::vec3(glm::make_vec4(
            &pixels[(static_cast<std::size_t>(y) * cube.faceSize + x) *
                    4]));
    };
    return glm::mix(
        glm::mix(sample(x0, y0), sample(x1, y0), tx),
        glm::mix(sample(x0, y1), sample(x1, y1), tx), ty);
}

float RadicalInverse(uint32_t bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) |
           ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) |
           ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) |
           ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) |
           ((bits & 0xFF00FF00u) >> 8u);
    return static_cast<float>(bits) * 2.3283064365386963e-10f;
}

glm::vec2 Hammersley(uint32_t index, uint32_t count) {
    return glm::vec2(static_cast<float>(index) / count,
                     RadicalInverse(index));
}

void BuildBasis(const glm::vec3& normal, glm::vec3& tangent,
                glm::vec3& bitangent) {
    const glm::vec3 up = std::abs(normal.y) < 0.999f
        ? glm::vec3(0.0f, 1.0f, 0.0f)
        : glm::vec3(1.0f, 0.0f, 0.0f);
    tangent = glm::normalize(glm::cross(up, normal));
    bitangent = glm::cross(normal, tangent);
}

glm::vec3 ImportanceSampleGGX(const glm::vec2& xi,
                              const glm::vec3& normal,
                              float roughness) {
    const float alpha = roughness * roughness;
    const float phi = 2.0f * Pi * xi.x;
    const float denominator =
        1.0f + (alpha * alpha - 1.0f) * xi.y;
    const float cosTheta = std::sqrt(
        std::max(0.0f, (1.0f - xi.y) / std::max(denominator, 1e-6f)));
    const float sinTheta = std::sqrt(
        std::max(0.0f, 1.0f - cosTheta * cosTheta));
    const glm::vec3 halfway(
        std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, cosTheta);
    glm::vec3 tangent;
    glm::vec3 bitangent;
    BuildBasis(normal, tangent, bitangent);
    return glm::normalize(
        tangent * halfway.x + bitangent * halfway.y + normal * halfway.z);
}

CubePixels BuildIrradiance(const CubePixels& source) {
    const uint32_t faceSize = 32;
    const uint32_t sampleCount = 256;
    CubePixels output;
    output.faceSize = faceSize;
    output.mipLevels = 1;
    output.levels.resize(6);
    for (uint32_t face = 0; face < 6; ++face) {
        std::vector<float>& pixels = output.levels[face];
        pixels.resize(static_cast<std::size_t>(faceSize) * faceSize * 4);
        for (uint32_t y = 0; y < faceSize; ++y) {
            for (uint32_t x = 0; x < faceSize; ++x) {
                const float u = 2.0f * (x + 0.5f) / faceSize - 1.0f;
                const float v = 2.0f * (y + 0.5f) / faceSize - 1.0f;
                const glm::vec3 normal = FaceDirection(face, u, v);
                glm::vec3 tangent;
                glm::vec3 bitangent;
                BuildBasis(normal, tangent, bitangent);
                glm::vec3 sum(0.0f);
                for (uint32_t sampleIndex = 0; sampleIndex < sampleCount;
                     ++sampleIndex) {
                    const glm::vec2 xi = Hammersley(
                        sampleIndex, sampleCount);
                    const float radius = std::sqrt(xi.x);
                    const float phi = 2.0f * Pi * xi.y;
                    const glm::vec3 local(
                        radius * std::cos(phi), radius * std::sin(phi),
                        std::sqrt(std::max(0.0f, 1.0f - xi.x)));
                    const glm::vec3 direction = glm::normalize(
                        tangent * local.x + bitangent * local.y +
                        normal * local.z);
                    sum += SampleCube(source, direction);
                }
                const glm::vec4 irradiance(
                    sum * (Pi / sampleCount), 1.0f);
                std::memcpy(
                    &pixels[(static_cast<std::size_t>(y) * faceSize + x) *
                            4],
                    &irradiance[0], sizeof(float) * 4);
            }
        }
    }
    return output;
}

CubePixels BuildPrefiltered(const CubePixels& source) {
    const uint32_t faceSize = 128;
    const uint32_t sampleCount = 128;
    CubePixels output;
    output.faceSize = faceSize;
    output.mipLevels = CalculateMipLevels(faceSize);
    output.levels.resize(static_cast<std::size_t>(output.mipLevels) * 6);
    uint32_t mipSize = faceSize;
    for (uint32_t mip = 0; mip < output.mipLevels; ++mip) {
        const float roughness = output.mipLevels > 1
            ? static_cast<float>(mip) / (output.mipLevels - 1)
            : 0.0f;
        for (uint32_t face = 0; face < 6; ++face) {
            std::vector<float>& pixels = output.levels[mip * 6 + face];
            pixels.resize(static_cast<std::size_t>(mipSize) * mipSize * 4);
            for (uint32_t y = 0; y < mipSize; ++y) {
                for (uint32_t x = 0; x < mipSize; ++x) {
                    const float u = 2.0f * (x + 0.5f) / mipSize - 1.0f;
                    const float v = 2.0f * (y + 0.5f) / mipSize - 1.0f;
                    const glm::vec3 normal = FaceDirection(face, u, v);
                    const glm::vec3 view = normal;
                    glm::vec3 sum(0.0f);
                    float weight = 0.0f;
                    if (roughness < 0.001f) {
                        sum = SampleCube(source, normal);
                        weight = 1.0f;
                    } else {
                        for (uint32_t sampleIndex = 0;
                             sampleIndex < sampleCount; ++sampleIndex) {
                            const glm::vec3 halfway = ImportanceSampleGGX(
                                Hammersley(sampleIndex, sampleCount), normal,
                                roughness);
                            const glm::vec3 light = glm::normalize(
                                2.0f * glm::dot(view, halfway) * halfway -
                                view);
                            const float nDotL =
                                std::max(glm::dot(normal, light), 0.0f);
                            if (nDotL > 0.0f) {
                                sum += SampleCube(source, light) * nDotL;
                                weight += nDotL;
                            }
                        }
                    }
                    const glm::vec4 color(
                        sum / std::max(weight, 1e-6f), 1.0f);
                    std::memcpy(
                        &pixels[(static_cast<std::size_t>(y) * mipSize + x) *
                                4],
                        &color[0], sizeof(float) * 4);
                }
            }
        }
        mipSize = std::max(1u, mipSize / 2);
    }
    return output;
}

float GeometrySchlickGGX(float nDotDirection, float roughness) {
    const float alpha = roughness * roughness;
    const float k = alpha * 0.5f;
    return nDotDirection /
        std::max(nDotDirection * (1.0f - k) + k, 1e-6f);
}

glm::vec2 IntegrateBrdf(float nDotV, float roughness) {
    const uint32_t sampleCount = 256;
    const glm::vec3 normal(0.0f, 0.0f, 1.0f);
    const glm::vec3 view(
        std::sqrt(std::max(0.0f, 1.0f - nDotV * nDotV)), 0.0f,
        nDotV);
    glm::vec2 result(0.0f);
    for (uint32_t sampleIndex = 0; sampleIndex < sampleCount;
         ++sampleIndex) {
        const glm::vec3 halfway = ImportanceSampleGGX(
            Hammersley(sampleIndex, sampleCount), normal, roughness);
        const glm::vec3 light = glm::normalize(
            2.0f * glm::dot(view, halfway) * halfway - view);
        const float nDotL = std::max(light.z, 0.0f);
        const float nDotH = std::max(halfway.z, 0.0f);
        const float vDotH = std::max(glm::dot(view, halfway), 0.0f);
        if (nDotL > 0.0f) {
            const float geometry =
                GeometrySchlickGGX(nDotV, roughness) *
                GeometrySchlickGGX(nDotL, roughness);
            const float geometryVisibility =
                geometry * vDotH /
                std::max(nDotH * nDotV, 1e-6f);
            const float fresnel = std::pow(1.0f - vDotH, 5.0f);
            result.x += (1.0f - fresnel) * geometryVisibility;
            result.y += fresnel * geometryVisibility;
        }
    }
    return result / static_cast<float>(sampleCount);
}

std::vector<std::vector<float>> BuildBrdfLut(uint32_t size) {
    std::vector<std::vector<float>> levels(1);
    levels[0].resize(static_cast<std::size_t>(size) * size * 4);
    for (uint32_t y = 0; y < size; ++y) {
        const float roughness = (y + 0.5f) / size;
        for (uint32_t x = 0; x < size; ++x) {
            const float nDotV = (x + 0.5f) / size;
            const glm::vec2 integrated = IntegrateBrdf(nDotV, roughness);
            const glm::vec4 value(integrated, 0.0f, 1.0f);
            std::memcpy(
                &levels[0][(static_cast<std::size_t>(y) * size + x) * 4],
                &value[0], sizeof(float) * 4);
        }
    }
    return levels;
}

void CreateImage(Device* device, uint32_t width, uint32_t height,
                 uint32_t mipLevels, uint32_t layers, bool cubeCompatible,
                 VkImage& image, VkDeviceMemory& memory) {
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.flags = cubeCompatible
        ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    imageInfo.extent = { width, height, 1 };
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = layers;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage =
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(device->GetVkDevice(), &imageInfo, nullptr, &image) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to create HDR cubemap image");
    }

    VkMemoryRequirements requirements;
    vkGetImageMemoryRequirements(device->GetVkDevice(), image, &requirements);
    VkMemoryAllocateInfo allocation = {};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = device->GetInstance()->GetMemoryTypeIndex(
        requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(device->GetVkDevice(), &allocation, nullptr,
                         &memory) != VK_SUCCESS) {
        vkDestroyImage(device->GetVkDevice(), image, nullptr);
        image = VK_NULL_HANDLE;
        throw std::runtime_error("Failed to allocate HDR cubemap memory");
    }
    vkBindImageMemory(device->GetVkDevice(), image, memory, 0);
}

void UploadTexture(
    Device* device, VkCommandPool commandPool, uint32_t width,
    uint32_t height, uint32_t mipLevels, uint32_t layers,
    bool cubeCompatible, const std::vector<std::vector<float>>& pixels,
    VkImage& image, VkDeviceMemory& memory, VkImageView& view,
    VkSampler& sampler) {
    std::size_t totalFloatCount = 0;
    for (const std::vector<float>& level : pixels) {
        totalFloatCount += level.size();
    }
    const VkDeviceSize totalBytes = static_cast<VkDeviceSize>(
        totalFloatCount * sizeof(float));
    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    BufferUtils::CreateBuffer(
        device, totalBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer, stagingMemory);

    std::vector<VkBufferImageCopy> regions;
    regions.reserve(pixels.size());
    void* mapped = nullptr;
    vkMapMemory(device->GetVkDevice(), stagingMemory, 0, totalBytes, 0,
                &mapped);
    std::size_t byteOffset = 0;
    uint32_t mipWidth = width;
    uint32_t mipHeight = height;
    for (uint32_t mip = 0; mip < mipLevels; ++mip) {
        for (uint32_t layer = 0; layer < layers; ++layer) {
            const std::vector<float>& level = pixels[mip * layers + layer];
            const std::size_t byteCount = level.size() * sizeof(float);
            std::memcpy(static_cast<unsigned char*>(mapped) + byteOffset,
                        level.data(), byteCount);
            VkBufferImageCopy region = {};
            region.bufferOffset = static_cast<VkDeviceSize>(byteOffset);
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = mip;
            region.imageSubresource.baseArrayLayer = layer;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = { mipWidth, mipHeight, 1 };
            regions.push_back(region);
            byteOffset += byteCount;
        }
        mipWidth = std::max(1u, mipWidth / 2);
        mipHeight = std::max(1u, mipHeight / 2);
    }
    vkUnmapMemory(device->GetVkDevice(), stagingMemory);

    CreateImage(device, width, height, mipLevels, layers, cubeCompatible,
                image, memory);
    VkCommandBufferAllocateInfo allocationInfo = {};
    allocationInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocationInfo.commandPool = commandPool;
    allocationInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocationInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device->GetVkDevice(), &allocationInfo,
                                 &commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate texture upload command");
    }
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.layerCount = layers;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);
    vkCmdCopyBufferToImage(
        commandBuffer, stagingBuffer, image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        static_cast<uint32_t>(regions.size()), regions.data());

    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &barrier);
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(device->GetQueue(QueueFlags::Graphics), 1, &submitInfo,
                  VK_NULL_HANDLE);
    vkQueueWaitIdle(device->GetQueue(QueueFlags::Graphics));
    vkFreeCommandBuffers(device->GetVkDevice(), commandPool, 1,
                         &commandBuffer);
    vkDestroyBuffer(device->GetVkDevice(), stagingBuffer, nullptr);
    vkFreeMemory(device->GetVkDevice(), stagingMemory, nullptr);

    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = cubeCompatible
        ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.layerCount = layers;
    if (vkCreateImageView(device->GetVkDevice(), &viewInfo, nullptr, &view) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to create environment image view");
    }

    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = static_cast<float>(mipLevels - 1);
    if (vkCreateSampler(device->GetVkDevice(), &samplerInfo, nullptr,
                        &sampler) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create environment sampler");
    }
}

} // namespace

Resources Create(Device* device, VkCommandPool graphicsCommandPool,
                 const std::string& hdrPath, uint32_t maximumFaceSize) {
    CubePixels background = BuildCubePixels(hdrPath, maximumFaceSize);
    Resources resources;
    resources.mipLevels = background.mipLevels;
    resources.hasHdr = !hdrPath.empty();

    CubePixels irradiance;
    CubePixels prefiltered;
    std::vector<std::vector<float>> brdfLut;
    uint32_t brdfLutSize = 1;
    if (resources.hasHdr) {
        irradiance = BuildIrradiance(background);
        prefiltered = BuildPrefiltered(background);
        brdfLutSize = 128;
        brdfLut = BuildBrdfLut(brdfLutSize);
    } else {
        irradiance = background;
        prefiltered = background;
        brdfLut = { { 1.0f, 0.0f, 0.0f, 1.0f } };
    }
    resources.prefilteredMipLevels = prefiltered.mipLevels;

    try {
        UploadTexture(
            device, graphicsCommandPool, background.faceSize,
            background.faceSize, background.mipLevels, 6, true,
            background.levels, resources.image, resources.memory,
            resources.view, resources.sampler);
        UploadTexture(
            device, graphicsCommandPool, irradiance.faceSize,
            irradiance.faceSize, irradiance.mipLevels, 6, true,
            irradiance.levels, resources.irradianceImage,
            resources.irradianceMemory, resources.irradianceView,
            resources.irradianceSampler);
        UploadTexture(
            device, graphicsCommandPool, prefiltered.faceSize,
            prefiltered.faceSize, prefiltered.mipLevels, 6, true,
            prefiltered.levels, resources.prefilteredImage,
            resources.prefilteredMemory, resources.prefilteredView,
            resources.prefilteredSampler);
        UploadTexture(
            device, graphicsCommandPool, brdfLutSize, brdfLutSize, 1, 1,
            false, brdfLut, resources.brdfLutImage,
            resources.brdfLutMemory, resources.brdfLutView,
            resources.brdfLutSampler);
    } catch (...) {
        Destroy(device, resources);
        throw;
    }
    return resources;
}
void Destroy(Device* device, Resources& resources) {
    if (resources.brdfLutSampler != VK_NULL_HANDLE) {
        vkDestroySampler(device->GetVkDevice(), resources.brdfLutSampler,
                         nullptr);
    }
    if (resources.brdfLutView != VK_NULL_HANDLE) {
        vkDestroyImageView(device->GetVkDevice(), resources.brdfLutView,
                           nullptr);
    }
    if (resources.brdfLutImage != VK_NULL_HANDLE) {
        vkDestroyImage(device->GetVkDevice(), resources.brdfLutImage,
                       nullptr);
    }
    if (resources.brdfLutMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device->GetVkDevice(), resources.brdfLutMemory,
                     nullptr);
    }
    if (resources.prefilteredSampler != VK_NULL_HANDLE) {
        vkDestroySampler(device->GetVkDevice(),
                         resources.prefilteredSampler, nullptr);
    }
    if (resources.prefilteredView != VK_NULL_HANDLE) {
        vkDestroyImageView(device->GetVkDevice(), resources.prefilteredView,
                           nullptr);
    }
    if (resources.prefilteredImage != VK_NULL_HANDLE) {
        vkDestroyImage(device->GetVkDevice(), resources.prefilteredImage,
                       nullptr);
    }
    if (resources.prefilteredMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device->GetVkDevice(), resources.prefilteredMemory,
                     nullptr);
    }
    if (resources.irradianceSampler != VK_NULL_HANDLE) {
        vkDestroySampler(device->GetVkDevice(), resources.irradianceSampler,
                         nullptr);
    }
    if (resources.irradianceView != VK_NULL_HANDLE) {
        vkDestroyImageView(device->GetVkDevice(), resources.irradianceView,
                           nullptr);
    }
    if (resources.irradianceImage != VK_NULL_HANDLE) {
        vkDestroyImage(device->GetVkDevice(), resources.irradianceImage,
                       nullptr);
    }
    if (resources.irradianceMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device->GetVkDevice(), resources.irradianceMemory,
                     nullptr);
    }
    if (resources.sampler != VK_NULL_HANDLE) {
        vkDestroySampler(device->GetVkDevice(), resources.sampler, nullptr);
    }
    if (resources.view != VK_NULL_HANDLE) {
        vkDestroyImageView(device->GetVkDevice(), resources.view, nullptr);
    }
    if (resources.image != VK_NULL_HANDLE) {
        vkDestroyImage(device->GetVkDevice(), resources.image, nullptr);
    }
    if (resources.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device->GetVkDevice(), resources.memory, nullptr);
    }
    resources = Resources();
}

} // namespace EnvironmentMap
