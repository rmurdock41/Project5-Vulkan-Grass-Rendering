#pragma once

#include "Device.h"

#include <string>

namespace EnvironmentMap {

struct Resources {
    // Full-resolution cubemap used for the visible sky.
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    uint32_t mipLevels = 1;

    // Diffuse cosine convolution.
    VkImage irradianceImage = VK_NULL_HANDLE;
    VkDeviceMemory irradianceMemory = VK_NULL_HANDLE;
    VkImageView irradianceView = VK_NULL_HANDLE;
    VkSampler irradianceSampler = VK_NULL_HANDLE;

    // GGX convolution; roughness is encoded across mip levels.
    VkImage prefilteredImage = VK_NULL_HANDLE;
    VkDeviceMemory prefilteredMemory = VK_NULL_HANDLE;
    VkImageView prefilteredView = VK_NULL_HANDLE;
    VkSampler prefilteredSampler = VK_NULL_HANDLE;
    uint32_t prefilteredMipLevels = 1;

    // Split-sum BRDF integration lookup table.
    VkImage brdfLutImage = VK_NULL_HANDLE;
    VkDeviceMemory brdfLutMemory = VK_NULL_HANDLE;
    VkImageView brdfLutView = VK_NULL_HANDLE;
    VkSampler brdfLutSampler = VK_NULL_HANDLE;

    bool hasHdr = false;
};

Resources Create(Device* device, VkCommandPool graphicsCommandPool,
                 const std::string& hdrPath, uint32_t maximumFaceSize = 1024);
void Destroy(Device* device, Resources& resources);

} // namespace EnvironmentMap
