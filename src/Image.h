#pragma once

#include <vulkan/vulkan.h>
#include "Device.h"

#include <cstddef>

namespace Image {

    void SetMipmapsEnabled(bool enabled);
    void Create(Device* device, uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory, uint32_t mipLevels = 1);
    void TransitionLayout(Device* device, VkCommandPool commandPool, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels = 1);
    VkImageView CreateView(Device* device, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels = 1);
    void CopyFromBuffer(Device* device, VkCommandPool commandPool, VkBuffer buffer, VkImage& image, uint32_t width, uint32_t height);
    void FromPixels(Device* device, VkCommandPool commandPool, const unsigned char* pixels, uint32_t width, uint32_t height, VkImage& image, VkDeviceMemory& imageMemory, VkFormat format = VK_FORMAT_R8G8B8A8_UNORM, uint32_t* mipLevelsOut = nullptr);
    void FromEncodedMemory(Device* device, VkCommandPool commandPool, const unsigned char* encodedBytes, std::size_t encodedByteCount, VkImage& image, VkDeviceMemory& imageMemory, VkFormat format = VK_FORMAT_R8G8B8A8_SRGB, uint32_t* mipLevelsOut = nullptr);
    void FromFile(Device* device, VkCommandPool commandPool, const char* path, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkImageLayout layout, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory, uint32_t* mipLevelsOut = nullptr);
}
