#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "Image.h"
#include "Device.h"
#include "Instance.h"
#include "BufferUtils.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {

bool mipmapsEnabled = true;

uint32_t CalculateMipLevels(uint32_t width, uint32_t height) {
    uint32_t largestDimension = std::max(width, height);
    uint32_t levels = 1;
    while (largestDimension > 1) {
        largestDimension /= 2;
        ++levels;
    }
    return levels;
}

bool SupportsLinearBlit(Device* device, VkFormat format) {
    VkFormatProperties properties = {};
    vkGetPhysicalDeviceFormatProperties(
        device->GetInstance()->GetPhysicalDevice(), format, &properties);
    const VkFormatFeatureFlags required =
        VK_FORMAT_FEATURE_BLIT_SRC_BIT |
        VK_FORMAT_FEATURE_BLIT_DST_BIT |
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    return (properties.optimalTilingFeatures & required) == required;
}

VkCommandBuffer BeginUploadCommands(Device* device,
                                    VkCommandPool commandPool) {
    VkCommandBufferAllocateInfo allocateInfo = {};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandPool = commandPool;
    allocateInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device->GetVkDevice(), &allocateInfo,
                                 &commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error(
            "Failed to allocate texture upload command buffer");
    }

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        vkFreeCommandBuffers(device->GetVkDevice(), commandPool, 1,
                             &commandBuffer);
        throw std::runtime_error(
            "Failed to begin texture upload command buffer");
    }
    return commandBuffer;
}

void EndUploadCommands(Device* device, VkCommandPool commandPool,
                       VkCommandBuffer commandBuffer) {
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        vkFreeCommandBuffers(device->GetVkDevice(), commandPool, 1,
                             &commandBuffer);
        throw std::runtime_error(
            "Failed to record texture upload command buffer");
    }

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    VkQueue graphicsQueue = device->GetQueue(QueueFlags::Graphics);
    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) !=
        VK_SUCCESS) {
        vkFreeCommandBuffers(device->GetVkDevice(), commandPool, 1,
                             &commandBuffer);
        throw std::runtime_error("Failed to submit texture upload");
    }
    vkQueueWaitIdle(graphicsQueue);
    vkFreeCommandBuffers(device->GetVkDevice(), commandPool, 1,
                         &commandBuffer);
}

void UploadPixelsAndGenerateMipmaps(
        Device* device, VkCommandPool commandPool, VkBuffer stagingBuffer,
        VkImage image, uint32_t width, uint32_t height,
        uint32_t mipLevels) {
    VkCommandBuffer commandBuffer =
        BeginUploadCommands(device, commandPool);

    VkImageMemoryBarrier allLevelsToDestination = {};
    allLevelsToDestination.sType =
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    allLevelsToDestination.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    allLevelsToDestination.newLayout =
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    allLevelsToDestination.srcQueueFamilyIndex =
        VK_QUEUE_FAMILY_IGNORED;
    allLevelsToDestination.dstQueueFamilyIndex =
        VK_QUEUE_FAMILY_IGNORED;
    allLevelsToDestination.image = image;
    allLevelsToDestination.subresourceRange.aspectMask =
        VK_IMAGE_ASPECT_COLOR_BIT;
    allLevelsToDestination.subresourceRange.baseMipLevel = 0;
    allLevelsToDestination.subresourceRange.levelCount = mipLevels;
    allLevelsToDestination.subresourceRange.baseArrayLayer = 0;
    allLevelsToDestination.subresourceRange.layerCount = 1;
    allLevelsToDestination.srcAccessMask = 0;
    allLevelsToDestination.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(
        commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
        &allLevelsToDestination);

    VkBufferImageCopy copy = {};
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.mipLevel = 0;
    copy.imageSubresource.baseArrayLayer = 0;
    copy.imageSubresource.layerCount = 1;
    copy.imageExtent = { width, height, 1 };
    vkCmdCopyBufferToImage(
        commandBuffer, stagingBuffer, image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

    int32_t mipWidth = static_cast<int32_t>(width);
    int32_t mipHeight = static_cast<int32_t>(height);
    for (uint32_t level = 1; level < mipLevels; ++level) {
        VkImageMemoryBarrier previousLevelToSource = {};
        previousLevelToSource.sType =
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        previousLevelToSource.oldLayout =
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        previousLevelToSource.newLayout =
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        previousLevelToSource.srcQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        previousLevelToSource.dstQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        previousLevelToSource.image = image;
        previousLevelToSource.subresourceRange.aspectMask =
            VK_IMAGE_ASPECT_COLOR_BIT;
        previousLevelToSource.subresourceRange.baseMipLevel = level - 1;
        previousLevelToSource.subresourceRange.levelCount = 1;
        previousLevelToSource.subresourceRange.baseArrayLayer = 0;
        previousLevelToSource.subresourceRange.layerCount = 1;
        previousLevelToSource.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        previousLevelToSource.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(
            commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
            &previousLevelToSource);

        const int32_t nextWidth = mipWidth > 1 ? mipWidth / 2 : 1;
        const int32_t nextHeight = mipHeight > 1 ? mipHeight / 2 : 1;
        VkImageBlit blit = {};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = level - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;
        blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = level;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;
        blit.dstOffsets[1] = { nextWidth, nextHeight, 1 };
        vkCmdBlitImage(
            commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
            VK_FILTER_LINEAR);

        VkImageMemoryBarrier previousLevelToShader =
            previousLevelToSource;
        previousLevelToShader.oldLayout =
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        previousLevelToShader.newLayout =
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        previousLevelToShader.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        previousLevelToShader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(
            commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
            nullptr, 1, &previousLevelToShader);

        mipWidth = nextWidth;
        mipHeight = nextHeight;
    }

    VkImageMemoryBarrier lastLevelToShader = {};
    lastLevelToShader.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    lastLevelToShader.oldLayout =
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    lastLevelToShader.newLayout =
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    lastLevelToShader.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    lastLevelToShader.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    lastLevelToShader.image = image;
    lastLevelToShader.subresourceRange.aspectMask =
        VK_IMAGE_ASPECT_COLOR_BIT;
    lastLevelToShader.subresourceRange.baseMipLevel = mipLevels - 1;
    lastLevelToShader.subresourceRange.levelCount = 1;
    lastLevelToShader.subresourceRange.baseArrayLayer = 0;
    lastLevelToShader.subresourceRange.layerCount = 1;
    lastLevelToShader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    lastLevelToShader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(
        commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
        1, &lastLevelToShader);

    EndUploadCommands(device, commandPool, commandBuffer);
}

} // namespace

void Image::SetMipmapsEnabled(bool enabled) {
    mipmapsEnabled = enabled;
}

void Image::Create(Device* device, uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory, uint32_t mipLevels) {
    // Create Vulkan image
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device->GetVkDevice(), &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image");
    }

    // Allocate memory for the image
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device->GetVkDevice(), image, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = device->GetInstance()->GetMemoryTypeIndex(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device->GetVkDevice(), &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate image memory");
    }

    // Bind the image
    vkBindImageMemory(device->GetVkDevice(), image, imageMemory, 0);
}

void Image::TransitionLayout(Device* device, VkCommandPool commandPool, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels) {
    auto hasStencilComponent = [](VkFormat format) {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
  };

    // Use an image memory barrier (type of pipeline barrier) to transition image layout
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
  
    if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    
        if (hasStencilComponent(format)) {
            barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
    }
    else {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }
  
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
  
    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;
  
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    } else {
        throw std::invalid_argument("Unsupported layout transition");
    }

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device->GetVkDevice(), &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);
  
    vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
  
    vkEndCommandBuffer(commandBuffer);
    
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(device->GetQueue(QueueFlags::Graphics), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(device->GetQueue(QueueFlags::Graphics));
    vkFreeCommandBuffers(device->GetVkDevice(), commandPool, 1, &commandBuffer);
}

VkImageView Image::CreateView(Device* device, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels) {
    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;

    // Describe the image's purpose and which part of the image should be accessed
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    if (vkCreateImageView(device->GetVkDevice(), &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
      throw std::runtime_error("Failed to texture image view");
    }

    return imageView;
}

void Image::CopyFromBuffer(Device* device, VkCommandPool commandPool, VkBuffer buffer, VkImage& image, uint32_t width, uint32_t height) {
    // Specify which part of the buffer is going to be copied to which part of the image
    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { width, height, 1 };

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device->GetVkDevice(), &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(device->GetQueue(QueueFlags::Transfer), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(device->GetQueue(QueueFlags::Transfer));
    vkFreeCommandBuffers(device->GetVkDevice(), commandPool, 1, &commandBuffer);
}

void Image::FromPixels(Device* device, VkCommandPool commandPool,
                       const unsigned char* pixels, uint32_t width,
                       uint32_t height, VkImage& image,
                       VkDeviceMemory& imageMemory, VkFormat format,
                       uint32_t* mipLevelsOut) {
    if (pixels == nullptr || width == 0 || height == 0) {
        throw std::invalid_argument("Invalid RGBA texture pixels");
    }

    const VkDeviceSize imageSize =
        static_cast<VkDeviceSize>(width) * height * 4;
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    BufferUtils::CreateBuffer(
        device, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer, stagingBufferMemory);

    void* data = nullptr;
    vkMapMemory(device->GetVkDevice(), stagingBufferMemory, 0, imageSize, 0,
                &data);
    std::memcpy(data, pixels, static_cast<std::size_t>(imageSize));
    vkUnmapMemory(device->GetVkDevice(), stagingBufferMemory);

    const uint32_t requestedMipLevels =
        mipmapsEnabled ? CalculateMipLevels(width, height) : 1;
    const uint32_t mipLevels =
        requestedMipLevels > 1 && SupportsLinearBlit(device, format)
            ? requestedMipLevels
            : 1;
    Image::Create(
        device, width, height, format, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, image, imageMemory, mipLevels);
    UploadPixelsAndGenerateMipmaps(
        device, commandPool, stagingBuffer, image, width, height,
        mipLevels);
    if (mipLevelsOut != nullptr) {
        *mipLevelsOut = mipLevels;
    }

    vkDestroyBuffer(device->GetVkDevice(), stagingBuffer, nullptr);
    vkFreeMemory(device->GetVkDevice(), stagingBufferMemory, nullptr);
}

void Image::FromEncodedMemory(Device* device, VkCommandPool commandPool,
                              const unsigned char* encodedBytes,
                              std::size_t encodedByteCount, VkImage& image,
                              VkDeviceMemory& imageMemory, VkFormat format,
                              uint32_t* mipLevelsOut) {
    if (encodedBytes == nullptr || encodedByteCount == 0 ||
        encodedByteCount > static_cast<std::size_t>(INT_MAX)) {
        throw std::invalid_argument("Invalid encoded texture data");
    }

    int width = 0;
    int height = 0;
    int channelCount = 0;
    stbi_uc* pixels = stbi_load_from_memory(
        encodedBytes, static_cast<int>(encodedByteCount), &width, &height,
        &channelCount, STBI_rgb_alpha);
    if (pixels == nullptr || width <= 0 || height <= 0) {
        throw std::runtime_error(std::string("Failed to decode embedded texture: ") +
                                 stbi_failure_reason());
    }

    try {
        FromPixels(device, commandPool, pixels, static_cast<uint32_t>(width),
                   static_cast<uint32_t>(height), image, imageMemory, format,
                   mipLevelsOut);
    } catch (...) {
        stbi_image_free(pixels);
        throw;
    }
    stbi_image_free(pixels);
}

void Image::FromFile(Device* device, VkCommandPool commandPool, const char* path, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkImageLayout layout, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory, uint32_t* mipLevelsOut) {
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(path, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (!pixels) {
        throw std::runtime_error("Failed to load texture image");
    }
    if (tiling != VK_IMAGE_TILING_OPTIMAL ||
        (usage & VK_IMAGE_USAGE_SAMPLED_BIT) == 0 ||
        layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ||
        (properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0) {
        stbi_image_free(pixels);
        throw std::invalid_argument(
            "Mipmapped file textures require optimal, sampled, "
            "shader-readable device-local images");
    }
    try {
        FromPixels(device, commandPool, pixels,
                   static_cast<uint32_t>(texWidth),
                   static_cast<uint32_t>(texHeight), image, imageMemory,
                   format, mipLevelsOut);
    } catch (...) {
        stbi_image_free(pixels);
        throw;
    }
    stbi_image_free(pixels);
}
