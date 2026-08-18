#include "Renderer.h"
#include "Instance.h"
#include "ShaderModule.h"
#include "Vertex.h"
#include "Blades.h"
#include "Camera.h"
#include "Image.h"
#include "Frustum.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include "BufferUtils.h"
#include <glm/gtc/matrix_transform.hpp>
#include <cstring>

static constexpr unsigned int WORKGROUP_SIZE = 32;

struct FlowerCullPushConstants {
    float density;
    std::uint32_t enabled;
};

struct FlowerRenderPushConstants {
    float heightScale;
};

namespace {

float ProjectedShadowExtentPixels(
    const glm::vec3& boundsMin, const glm::vec3& boundsMax,
    const glm::mat4& lightViewProjection, uint32_t shadowMapSize) {
    glm::vec2 projectedMin(std::numeric_limits<float>::max());
    glm::vec2 projectedMax(-std::numeric_limits<float>::max());
    bool valid = false;
    for (uint32_t corner = 0; corner < 8; ++corner) {
        const glm::vec3 worldCorner(
            (corner & 1) ? boundsMax.x : boundsMin.x,
            (corner & 2) ? boundsMax.y : boundsMin.y,
            (corner & 4) ? boundsMax.z : boundsMin.z);
        const glm::vec4 clip = lightViewProjection *
            glm::vec4(worldCorner, 1.0f);
        if (std::abs(clip.w) < 1.0e-6f) {
            continue;
        }
        const glm::vec2 ndc = glm::vec2(clip) / clip.w;
        projectedMin = glm::min(projectedMin, ndc);
        projectedMax = glm::max(projectedMax, ndc);
        valid = true;
    }
    if (!valid) {
        return std::numeric_limits<float>::max();
    }
    const glm::vec2 pixelExtent =
        (projectedMax - projectedMin) *
        (0.5f * static_cast<float>(shadowMapSize));
    return glm::max(pixelExtent.x, pixelExtent.y);
}

} // namespace

Renderer::Renderer(Device* device, SwapChain* swapChain, Scene* scene,
                   Camera* camera,
                   const EnvironmentSettings& environmentSettings,
                   bool frustumCullingEnabled,
                   bool shadowCullingEnabled)
  : device(device),
    logicalDevice(device->GetVkDevice()),
    swapChain(swapChain),
    scene(scene),
    camera(camera),
    environmentSettings(environmentSettings),
    frustumCullingEnabled(frustumCullingEnabled),
    shadowCullingEnabled(shadowCullingEnabled) {

    CreateCommandPools();
    CreateRenderPass();
    CreateShadowRenderPass();
    CreateCameraDescriptorSetLayout();
    CreateModelDescriptorSetLayout();
    CreateShadowDescriptorSetLayout();
    CreateEnvironmentDescriptorSetLayout();
    CreateTimeDescriptorSetLayout();
    CreateComputeDescriptorSetLayout();
    CreateShadowResources();
    CreateEnvironmentResources();
    CreateDescriptorPool();
    CreateCameraDescriptorSet();
    CreateModelDescriptorSets();
    CreateShadowDescriptorSet();
    CreateEnvironmentDescriptorSet();
    CreateGrassDescriptorSets();
    CreateTimeDescriptorSet();
    CreateComputeDescriptorSets();
    CreateFrameResources();
    CreateShadowPipeline();
    CreateGraphicsPipeline();
    CreateGrassPipeline();
    CreateFlowerPipeline();
    CreateComputePipeline();

    CreateSkyboxResources();
    CreateSkyboxPipeline();

    RecordCommandBuffers();
    RecordComputeCommandBuffers();
}

void Renderer::CreateShadowRenderPass() {
    shadowDepthFormat = device->GetInstance()->GetSupportedFormat(
        { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM },
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT);

    VkAttachmentDescription depthAttachment = {};
    depthAttachment.format = shadowDepthFormat;
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    // Every shadow pass clears the entire map, so previous contents are not
    // preserved. The render pass transitions it back to shader-readable.
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkAttachmentReference depthReference = {};
    depthReference.attachment = 0;
    depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 0;
    subpass.pDepthStencilAttachment = &depthReference;

    std::array<VkSubpassDependency, 2> dependencies = {};
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[0].dstStageMask =
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[0].dstAccessMask =
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask =
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].srcAccessMask =
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &depthAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount =
        static_cast<uint32_t>(dependencies.size());
    renderPassInfo.pDependencies = dependencies.data();

    if (vkCreateRenderPass(logicalDevice, &renderPassInfo, nullptr,
                           &shadowRenderPass) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shadow render pass");
    }
}

void Renderer::CreateCommandPools() {
    VkCommandPoolCreateInfo graphicsPoolInfo = {};
    graphicsPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    graphicsPoolInfo.queueFamilyIndex = device->GetInstance()->GetQueueFamilyIndices()[QueueFlags::Graphics];
    graphicsPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(logicalDevice, &graphicsPoolInfo, nullptr, &graphicsCommandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }

    VkCommandPoolCreateInfo computePoolInfo = {};
    computePoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    computePoolInfo.queueFamilyIndex = device->GetInstance()->GetQueueFamilyIndices()[QueueFlags::Compute];
    computePoolInfo.flags = 0;

    if (vkCreateCommandPool(logicalDevice, &computePoolInfo, nullptr, &computeCommandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }
}

void Renderer::CreateRenderPass() {
    // Color buffer attachment represented by one of the images from the swap chain
    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format = swapChain->GetVkImageFormat();
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    // Create a color attachment reference to be used with subpass
    VkAttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // Depth buffer attachment
    VkFormat depthFormat = device->GetInstance()->GetSupportedFormat({ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT }, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    VkAttachmentDescription depthAttachment = {};
    depthAttachment.format = depthFormat;
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // Create a depth attachment reference
    VkAttachmentReference depthAttachmentRef = {};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // Create subpass description
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };

    // Specify subpass dependency
    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    // Create render pass
    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(logicalDevice, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create render pass");
    }
}

void Renderer::CreateCameraDescriptorSetLayout() {
    // Describe the binding of the descriptor set layout
    VkDescriptorSetLayoutBinding uboLayoutBinding = {};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_ALL;
    uboLayoutBinding.pImmutableSamplers = nullptr;

    std::vector<VkDescriptorSetLayoutBinding> bindings = { uboLayoutBinding };

    // Create the descriptor set layout
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(logicalDevice, &layoutInfo, nullptr, &cameraDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }
}

void Renderer::CreateModelDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding uboLayoutBinding = {};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    uboLayoutBinding.pImmutableSamplers = nullptr;

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back(uboLayoutBinding);
    for (uint32_t bindingIndex = 1; bindingIndex <= 5; ++bindingIndex) {
        VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
        samplerLayoutBinding.binding = bindingIndex;
        samplerLayoutBinding.descriptorType =
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        samplerLayoutBinding.pImmutableSamplers = nullptr;
        bindings.push_back(samplerLayoutBinding);
    }

    // Create the descriptor set layout
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(logicalDevice, &layoutInfo, nullptr, &modelDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }
}

void Renderer::CreateShadowDescriptorSetLayout() {
    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(logicalDevice, &layoutInfo, nullptr,
                                    &shadowDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error(
            "Failed to create shadow descriptor set layout");
    }
}

void Renderer::CreateEnvironmentDescriptorSetLayout() {
    std::array<VkDescriptorSetLayoutBinding, 5> bindings = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    for (uint32_t binding = 2; binding <= 4; ++binding) {
        bindings[binding].binding = binding;
        bindings[binding].descriptorType =
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[binding].descriptorCount = 1;
        bindings[binding].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(
            logicalDevice, &layoutInfo, nullptr,
            &environmentDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error(
            "Failed to create environment descriptor set layout");
    }
}

void Renderer::CreateShadowResources() {
    Image::Create(
        device, ShadowMapSize, ShadowMapSize, shadowDepthFormat,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, shadowDepthImage,
        shadowDepthImageMemory);
    shadowDepthImageView = Image::CreateView(
        device, shadowDepthImage, shadowDepthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);

    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    // PCF is performed explicitly in the shader, keeping this compatible with
    // depth formats that do not support linear filtering.
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    samplerInfo.compareEnable = VK_TRUE;
    samplerInfo.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;
    if (vkCreateSampler(logicalDevice, &samplerInfo, nullptr,
                        &shadowSampler) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shadow sampler");
    }

    VkFramebufferCreateInfo framebufferInfo = {};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = shadowRenderPass;
    framebufferInfo.attachmentCount = 1;
    framebufferInfo.pAttachments = &shadowDepthImageView;
    framebufferInfo.width = ShadowMapSize;
    framebufferInfo.height = ShadowMapSize;
    framebufferInfo.layers = 1;
    if (vkCreateFramebuffer(logicalDevice, &framebufferInfo, nullptr,
                            &shadowFramebuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shadow framebuffer");
    }

    const glm::vec3 target(0.0f, 1.0f, 0.0f);
    const glm::vec3 lightDirection = glm::normalize(
        glm::vec3(-0.35f, 0.85f, 0.40f));
    const glm::vec3 lightPosition = target + lightDirection * 18.0f;
    glm::mat4 projection = glm::orthoRH_ZO(
        -10.0f, 10.0f, -10.0f, 10.0f, 0.1f, 40.0f);
    projection[1][1] *= -1.0f;
    shadowBufferObject.lightViewProjection = projection * glm::lookAtRH(
        lightPosition, target, glm::vec3(0.0f, 1.0f, 0.0f));
    shadowBufferObject.lightDirection = glm::vec4(lightDirection, 0.0f);

    BufferUtils::CreateBuffer(
        device, sizeof(ShadowBufferObject),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        shadowBuffer, shadowBufferMemory);
    void* mapped = nullptr;
    vkMapMemory(logicalDevice, shadowBufferMemory, 0,
                sizeof(ShadowBufferObject), 0, &mapped);
    std::memcpy(mapped, &shadowBufferObject, sizeof(ShadowBufferObject));
    vkUnmapMemory(logicalDevice, shadowBufferMemory);
}

void Renderer::DestroyShadowResources() {
    vkDestroyFramebuffer(logicalDevice, shadowFramebuffer, nullptr);
    vkDestroySampler(logicalDevice, shadowSampler, nullptr);
    vkDestroyImageView(logicalDevice, shadowDepthImageView, nullptr);
    vkDestroyImage(logicalDevice, shadowDepthImage, nullptr);
    vkFreeMemory(logicalDevice, shadowDepthImageMemory, nullptr);
    vkDestroyBuffer(logicalDevice, shadowBuffer, nullptr);
    vkFreeMemory(logicalDevice, shadowBufferMemory, nullptr);
}

void Renderer::CreateEnvironmentResources() {
    environmentMap = EnvironmentMap::Create(
        device, graphicsCommandPool, environmentSettings.path);
    environmentBufferObject.parameters = glm::vec4(
        environmentSettings.backgroundIntensity,
        environmentSettings.lightingIntensity,
        environmentSettings.rotationRadians,
        environmentMap.hasHdr ? 1.0f : 0.0f);
    environmentBufferObject.options = glm::vec4(
        environmentSettings.visible ? 1.0f : 0.0f,
        static_cast<float>(environmentMap.prefilteredMipLevels - 1),
        environmentSettings.directLightingIntensity, 0.0f);

    BufferUtils::CreateBuffer(
        device, sizeof(EnvironmentBufferObject),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        environmentBuffer, environmentBufferMemory);
    void* mapped = nullptr;
    vkMapMemory(logicalDevice, environmentBufferMemory, 0,
                sizeof(EnvironmentBufferObject), 0, &mapped);
    std::memcpy(mapped, &environmentBufferObject,
                sizeof(EnvironmentBufferObject));
    vkUnmapMemory(logicalDevice, environmentBufferMemory);
}

void Renderer::DestroyEnvironmentResources() {
    vkDestroyBuffer(logicalDevice, environmentBuffer, nullptr);
    vkFreeMemory(logicalDevice, environmentBufferMemory, nullptr);
    EnvironmentMap::Destroy(device, environmentMap);
}

void Renderer::SetEnvironmentRotationDegrees(float rotationDegrees) {
    rotationDegrees = std::fmod(rotationDegrees, 360.0f);
    if (rotationDegrees < 0.0f) {
        rotationDegrees += 360.0f;
    }
    environmentSettings.rotationRadians = glm::radians(rotationDegrees);
    environmentBufferObject.parameters.z =
        environmentSettings.rotationRadians;

    void* mapped = nullptr;
    if (vkMapMemory(logicalDevice, environmentBufferMemory, 0,
                    sizeof(EnvironmentBufferObject), 0, &mapped) !=
        VK_SUCCESS) {
        throw std::runtime_error(
            "Failed to map environment rotation buffer");
    }
    std::memcpy(mapped, &environmentBufferObject,
                sizeof(EnvironmentBufferObject));
    vkUnmapMemory(logicalDevice, environmentBufferMemory);
}

void Renderer::CreateTimeDescriptorSetLayout() {
    // Describe the binding of the descriptor set layout
    VkDescriptorSetLayoutBinding uboLayoutBinding = {};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    uboLayoutBinding.pImmutableSamplers = nullptr;

    std::vector<VkDescriptorSetLayoutBinding> bindings = { uboLayoutBinding };

    // Create the descriptor set layout
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(logicalDevice, &layoutInfo, nullptr, &timeDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }
}

void Renderer::CreateComputeDescriptorSetLayout() {
    // TODO: Create the descriptor set layout for the compute pipeline
    // Remember this is like a class definition stating why types of information
    // will be stored at each binding
    
    
    // Binding 0: Input blades (storage buffer)
    VkDescriptorSetLayoutBinding inputBladesBinding = {};
    inputBladesBinding.binding = 0;
    inputBladesBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    inputBladesBinding.descriptorCount = 1;
    inputBladesBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // Binding 1: Culled blades (storage buffer)
    VkDescriptorSetLayoutBinding culledBladesBinding = {};
    culledBladesBinding.binding = 1;
    culledBladesBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    culledBladesBinding.descriptorCount = 1;
    culledBladesBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // Binding 2: Num blades (storage buffer)
    VkDescriptorSetLayoutBinding numBladesBinding = {};
    numBladesBinding.binding = 2;
    numBladesBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    numBladesBinding.descriptorCount = 1;
    numBladesBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding culledFlowersBinding = {};
    culledFlowersBinding.binding = 3;
    culledFlowersBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    culledFlowersBinding.descriptorCount = 1;
    culledFlowersBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding flowerDrawBinding = {};
    flowerDrawBinding.binding = 4;
    flowerDrawBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    flowerDrawBinding.descriptorCount = 1;
    flowerDrawBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        inputBladesBinding,
        culledBladesBinding,
        numBladesBinding,
        culledFlowersBinding,
        flowerDrawBinding
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(logicalDevice, &layoutInfo, nullptr, &computeDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute descriptor set layout");
    }
}

void Renderer::CreateDescriptorPool() {
    // Describe which descriptor types that the descriptor sets will contain
    const uint32_t modelCount = static_cast<uint32_t>(scene->GetModels().size());
    const uint32_t bladeCount = static_cast<uint32_t>(scene->GetBlades().size());
    const uint32_t cameraCount = swapChain->GetCount();
    std::vector<VkDescriptorPoolSize> poolSizes = {
        // Per-frame cameras + time + shadow + environment + one model matrix per
        // mesh/blade group.
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          cameraCount + 3 + modelCount + bladeCount },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          5 + 5 * modelCount + bladeCount }
    };
    if (bladeCount > 0) {
        poolSizes.push_back(
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 * bladeCount });
    }
    
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = cameraCount + 3 + modelCount + 2 * bladeCount;

    if (vkCreateDescriptorPool(logicalDevice, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool");
    }
}

void Renderer::CreateShadowDescriptorSet() {
    VkDescriptorSetAllocateInfo allocateInfo = {};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = descriptorPool;
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = &shadowDescriptorSetLayout;
    if (vkAllocateDescriptorSets(logicalDevice, &allocateInfo,
                                 &shadowDescriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate shadow descriptor set");
    }

    VkDescriptorBufferInfo bufferInfo = {};
    bufferInfo.buffer = shadowBuffer;
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(ShadowBufferObject);

    VkDescriptorImageInfo imageInfo = {};
    imageInfo.sampler = shadowSampler;
    imageInfo.imageView = shadowDepthImageView;
    imageInfo.imageLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    std::array<VkWriteDescriptorSet, 2> writes = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = shadowDescriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].descriptorCount = 1;
    writes[0].pBufferInfo = &bufferInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = shadowDescriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(logicalDevice,
        static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void Renderer::CreateEnvironmentDescriptorSet() {
    VkDescriptorSetAllocateInfo allocateInfo = {};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = descriptorPool;
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = &environmentDescriptorSetLayout;
    if (vkAllocateDescriptorSets(logicalDevice, &allocateInfo,
                                 &environmentDescriptorSet) != VK_SUCCESS) {
        throw std::runtime_error(
            "Failed to allocate environment descriptor set");
    }

    std::array<VkDescriptorImageInfo, 4> imageInfos = {};
    imageInfos[0].sampler = environmentMap.sampler;
    imageInfos[0].imageView = environmentMap.view;
    imageInfos[1].sampler = environmentMap.irradianceSampler;
    imageInfos[1].imageView = environmentMap.irradianceView;
    imageInfos[2].sampler = environmentMap.prefilteredSampler;
    imageInfos[2].imageView = environmentMap.prefilteredView;
    imageInfos[3].sampler = environmentMap.brdfLutSampler;
    imageInfos[3].imageView = environmentMap.brdfLutView;
    for (VkDescriptorImageInfo& imageInfo : imageInfos) {
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    VkDescriptorBufferInfo bufferInfo = {};
    bufferInfo.buffer = environmentBuffer;
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(EnvironmentBufferObject);

    std::array<VkWriteDescriptorSet, 5> writes = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = environmentDescriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo = &imageInfos[0];

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = environmentDescriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[1].descriptorCount = 1;
    writes[1].pBufferInfo = &bufferInfo;

    for (uint32_t writeIndex = 2; writeIndex <= 4; ++writeIndex) {
        writes[writeIndex].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[writeIndex].dstSet = environmentDescriptorSet;
        writes[writeIndex].dstBinding = writeIndex;
        writes[writeIndex].descriptorType =
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[writeIndex].descriptorCount = 1;
        writes[writeIndex].pImageInfo = &imageInfos[writeIndex - 1];
    }

    vkUpdateDescriptorSets(logicalDevice,
        static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void Renderer::CreateCameraDescriptorSet() {
    const uint32_t frameCount = swapChain->GetCount();
    if (camera->GetBufferCount() != frameCount) {
        throw std::runtime_error(
            "Camera buffer count does not match swapchain image count");
    }
    cameraDescriptorSets.resize(frameCount);
    std::vector<VkDescriptorSetLayout> layouts(
        frameCount, cameraDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = frameCount;
    allocInfo.pSetLayouts = layouts.data();

    if (vkAllocateDescriptorSets(logicalDevice, &allocInfo,
                                 cameraDescriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate camera descriptor sets");
    }

    std::vector<VkDescriptorBufferInfo> bufferInfos(frameCount);
    std::vector<VkWriteDescriptorSet> writes(frameCount);
    for (uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
        bufferInfos[frameIndex].buffer = camera->GetBuffer(frameIndex);
        bufferInfos[frameIndex].offset = 0;
        bufferInfos[frameIndex].range = sizeof(CameraBufferObject);

        writes[frameIndex].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[frameIndex].dstSet = cameraDescriptorSets[frameIndex];
        writes[frameIndex].dstBinding = 0;
        writes[frameIndex].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[frameIndex].descriptorCount = 1;
        writes[frameIndex].pBufferInfo = &bufferInfos[frameIndex];
    }
    vkUpdateDescriptorSets(logicalDevice, frameCount, writes.data(), 0,
                           nullptr);
}

void Renderer::CreateModelDescriptorSets() {
    modelDescriptorSets.resize(scene->GetModels().size());

    // Describe the desciptor set
    std::vector<VkDescriptorSetLayout> layouts(
        modelDescriptorSets.size(), modelDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(modelDescriptorSets.size());
    allocInfo.pSetLayouts = layouts.data();

    // Allocate descriptor sets
    if (vkAllocateDescriptorSets(logicalDevice, &allocInfo, modelDescriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set");
    }

    std::vector<VkDescriptorBufferInfo> modelBufferInfos(modelDescriptorSets.size());
    const std::size_t materialTextureCount =
        static_cast<std::size_t>(MaterialTextureSlot::Count);
    std::vector<VkDescriptorImageInfo> imageInfos(
        modelDescriptorSets.size() * materialTextureCount);
    std::vector<VkWriteDescriptorSet> descriptorWrites(
        modelDescriptorSets.size() * (1 + materialTextureCount));

    for (uint32_t i = 0; i < scene->GetModels().size(); ++i) {
        modelBufferInfos[i].buffer = scene->GetModels()[i]->GetModelBuffer();
        modelBufferInfos[i].offset = 0;
        modelBufferInfos[i].range = sizeof(ModelBufferObject);

        const std::size_t writeOffset = i * (1 + materialTextureCount);
        descriptorWrites[writeOffset].sType =
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[writeOffset].dstSet = modelDescriptorSets[i];
        descriptorWrites[writeOffset].dstBinding = 0;
        descriptorWrites[writeOffset].dstArrayElement = 0;
        descriptorWrites[writeOffset].descriptorType =
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[writeOffset].descriptorCount = 1;
        descriptorWrites[writeOffset].pBufferInfo = &modelBufferInfos[i];

        for (std::size_t textureIndex = 0;
             textureIndex < materialTextureCount; ++textureIndex) {
            const std::size_t imageInfoIndex =
                i * materialTextureCount + textureIndex;
            const MaterialTextureSlot slot =
                static_cast<MaterialTextureSlot>(textureIndex);
            imageInfos[imageInfoIndex].imageLayout =
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfos[imageInfoIndex].imageView =
                scene->GetModels()[i]->GetTextureView(slot);
            imageInfos[imageInfoIndex].sampler =
                scene->GetModels()[i]->GetTextureSampler(slot);

            VkWriteDescriptorSet& write =
                descriptorWrites[writeOffset + 1 + textureIndex];
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = modelDescriptorSets[i];
            write.dstBinding = static_cast<uint32_t>(textureIndex + 1);
            write.dstArrayElement = 0;
            write.descriptorType =
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.descriptorCount = 1;
            write.pImageInfo = &imageInfos[imageInfoIndex];
        }
    }

    // Update descriptor sets
    vkUpdateDescriptorSets(logicalDevice, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
}

void Renderer::CreateGrassDescriptorSets() {
    // TODO: Create Descriptor sets for the grass.
    // This should involve creating descriptor sets which point to the model matrix of each group of grass blades

    grassDescriptorSets.resize(scene->GetBlades().size());
    if (grassDescriptorSets.empty()) {
        return;
    }

    // Describe the descriptor set
    VkDescriptorSetLayout layouts[] = { modelDescriptorSetLayout };
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(grassDescriptorSets.size());
    allocInfo.pSetLayouts = layouts;

    // Allocate descriptor sets
    if (vkAllocateDescriptorSets(logicalDevice, &allocInfo, grassDescriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate grass descriptor set");
    }

    std::vector<VkWriteDescriptorSet> descriptorWrites(2 * grassDescriptorSets.size());

    for (uint32_t i = 0; i < scene->GetBlades().size(); ++i) {
        VkDescriptorBufferInfo modelBufferInfo = {};
        modelBufferInfo.buffer = scene->GetBlades()[i]->GetModelBuffer();
        modelBufferInfo.offset = 0;
        modelBufferInfo.range = sizeof(ModelBufferObject);

      
        VkDescriptorImageInfo imageInfo = {};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = scene->GetModels()[0]->GetTextureView();
        imageInfo.sampler = scene->GetModels()[0]->GetTextureSampler();

        descriptorWrites[2 * i + 0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2 * i + 0].dstSet = grassDescriptorSets[i];
        descriptorWrites[2 * i + 0].dstBinding = 0;
        descriptorWrites[2 * i + 0].dstArrayElement = 0;
        descriptorWrites[2 * i + 0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[2 * i + 0].descriptorCount = 1;
        descriptorWrites[2 * i + 0].pBufferInfo = &modelBufferInfo;

        descriptorWrites[2 * i + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2 * i + 1].dstSet = grassDescriptorSets[i];
        descriptorWrites[2 * i + 1].dstBinding = 1;
        descriptorWrites[2 * i + 1].dstArrayElement = 0;
        descriptorWrites[2 * i + 1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[2 * i + 1].descriptorCount = 1;
        descriptorWrites[2 * i + 1].pImageInfo = &imageInfo;
    }

    vkUpdateDescriptorSets(logicalDevice, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
}

void Renderer::CreateTimeDescriptorSet() {
    // Describe the desciptor set
    VkDescriptorSetLayout layouts[] = { timeDescriptorSetLayout };
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = layouts;

    // Allocate descriptor sets
    if (vkAllocateDescriptorSets(logicalDevice, &allocInfo, &timeDescriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set");
    }

    // Configure the descriptors to refer to buffers
    VkDescriptorBufferInfo timeBufferInfo = {};
    timeBufferInfo.buffer = scene->GetTimeBuffer();
    timeBufferInfo.offset = 0;
    timeBufferInfo.range = sizeof(Time);

    std::array<VkWriteDescriptorSet, 1> descriptorWrites = {};
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = timeDescriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &timeBufferInfo;
    descriptorWrites[0].pImageInfo = nullptr;
    descriptorWrites[0].pTexelBufferView = nullptr;

    // Update descriptor sets
    vkUpdateDescriptorSets(logicalDevice, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
}

void Renderer::CreateComputeDescriptorSets() {
    // TODO: Create Descriptor sets for the compute pipeline
    // The descriptors should point to Storage buffers which will hold the grass blades, the culled grass blades, and the output number of grass blades 

    computeDescriptorSets.resize(scene->GetBlades().size());
    if (computeDescriptorSets.empty()) {
        return;
    }

    std::vector<VkDescriptorSetLayout> layouts(scene->GetBlades().size(), computeDescriptorSetLayout);

    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(computeDescriptorSets.size());
    allocInfo.pSetLayouts = layouts.data();

    if (vkAllocateDescriptorSets(logicalDevice, &allocInfo, computeDescriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate compute descriptor sets");
    }

    for (size_t i = 0; i < scene->GetBlades().size(); ++i) {
        VkDescriptorBufferInfo inputBladesInfo = {};
        inputBladesInfo.buffer = scene->GetBlades()[i]->GetBladesBuffer();
        inputBladesInfo.offset = 0;
        inputBladesInfo.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo culledBladesInfo = {};
        culledBladesInfo.buffer = scene->GetBlades()[i]->GetCulledBladesBuffer();
        culledBladesInfo.offset = 0;
        culledBladesInfo.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo numBladesInfo = {};
        numBladesInfo.buffer = scene->GetBlades()[i]->GetNumBladesBuffer();
        numBladesInfo.offset = 0;
        numBladesInfo.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo culledFlowersInfo = {};
        culledFlowersInfo.buffer =
            scene->GetBlades()[i]->GetCulledFlowersBuffer();
        culledFlowersInfo.offset = 0;
        culledFlowersInfo.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo flowerDrawInfo = {};
        flowerDrawInfo.buffer =
            scene->GetBlades()[i]->GetFlowerDrawIndirectBuffer();
        flowerDrawInfo.offset = 0;
        flowerDrawInfo.range = VK_WHOLE_SIZE;

        std::array<VkWriteDescriptorSet, 5> descriptorWrites = {};

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = computeDescriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &inputBladesInfo;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = computeDescriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pBufferInfo = &culledBladesInfo;

        descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2].dstSet = computeDescriptorSets[i];
        descriptorWrites[2].dstBinding = 2;
        descriptorWrites[2].dstArrayElement = 0;
        descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].pBufferInfo = &numBladesInfo;

        descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[3].dstSet = computeDescriptorSets[i];
        descriptorWrites[3].dstBinding = 3;
        descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[3].descriptorCount = 1;
        descriptorWrites[3].pBufferInfo = &culledFlowersInfo;

        descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[4].dstSet = computeDescriptorSets[i];
        descriptorWrites[4].dstBinding = 4;
        descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[4].descriptorCount = 1;
        descriptorWrites[4].pBufferInfo = &flowerDrawInfo;

        vkUpdateDescriptorSets(logicalDevice, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void Renderer::CreateGraphicsPipeline() {
    VkShaderModule vertShaderModule = ShaderModule::Create("shaders/graphics.vert.spv", logicalDevice);
    VkShaderModule fragShaderModule = ShaderModule::Create("shaders/graphics.frag.spv", logicalDevice);

    // Assign each shader module to the appropriate stage in the pipeline
    VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    // --- Set up fixed-function stages ---

    // Vertex input
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    // Input assembly
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // Viewports and Scissors (rectangles that define in which regions pixels are stored)
    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(swapChain->GetVkExtent().width);
    viewport.height = static_cast<float>(swapChain->GetVkExtent().height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = { 0, 0 };
    scissor.extent = swapChain->GetVkExtent();

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    // Rasterizer. Material-specific culling is selected below when the six
    // ordinary-model pipelines are created.
    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f;
    rasterizer.depthBiasClamp = 0.0f;
    rasterizer.depthBiasSlopeFactor = 0.0f;

    // Multisampling (turned off here)
    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f;
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;

    // Depth testing
    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f;
    depthStencil.maxDepthBounds = 1.0f;
    depthStencil.stencilTestEnable = VK_FALSE;

    // Opaque and alpha-mask materials do not blend. Alpha-blended materials
    // get their own state below and are rendered after the depth-writing
    // queues.
    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    // --> Global color blending settings
    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    std::vector<VkDescriptorSetLayout> descriptorSetLayouts = {
        cameraDescriptorSetLayout,
        modelDescriptorSetLayout,
        shadowDescriptorSetLayout,
        environmentDescriptorSetLayout
    };

    // Pipeline layout: used to specify uniform values
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
    pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = 0;

    if (vkCreatePipelineLayout(logicalDevice, &pipelineLayoutInfo, nullptr, &graphicsPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout");
    }

    // --- Create material-specific graphics pipelines ---
    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = nullptr;
    pipelineInfo.layout = graphicsPipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    const auto createMaterialPipeline =
        [&](VkPipeline& pipeline, bool blend, bool depthWrite,
            bool doubleSided, const char* label) {
            rasterizer.cullMode = doubleSided
                ? VK_CULL_MODE_NONE
                : VK_CULL_MODE_BACK_BIT;
            rasterizer.depthBiasEnable = VK_FALSE;
            rasterizer.depthBiasConstantFactor = 0.0f;
            rasterizer.depthBiasSlopeFactor = 0.0f;
            depthStencil.depthWriteEnable = depthWrite ? VK_TRUE : VK_FALSE;
            colorBlendAttachment.blendEnable = blend ? VK_TRUE : VK_FALSE;
            if (vkCreateGraphicsPipelines(
                    logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo,
                    nullptr, &pipeline) != VK_SUCCESS) {
                throw std::runtime_error(
                    std::string("Failed to create ") + label +
                    " graphics pipeline");
            }
        };

    createMaterialPipeline(graphicsOpaquePipeline, false, true, false,
                           "opaque single-sided");
    createMaterialPipeline(graphicsOpaqueDoubleSidedPipeline, false, true,
                           true, "opaque double-sided");
    createMaterialPipeline(graphicsMaskPipeline, false, true, false,
                           "alpha-mask single-sided");
    createMaterialPipeline(graphicsMaskDoubleSidedPipeline, false, true,
                           true, "alpha-mask double-sided");
    createMaterialPipeline(graphicsBlendPipeline, true, false, false,
                           "alpha-blend single-sided");
    createMaterialPipeline(graphicsBlendDoubleSidedPipeline, true, false,
                           true, "alpha-blend double-sided");

    vkDestroyShaderModule(logicalDevice, vertShaderModule, nullptr);
    vkDestroyShaderModule(logicalDevice, fragShaderModule, nullptr);
}

void Renderer::CreateShadowPipeline() {
    VkShaderModule vertShaderModule =
        ShaderModule::Create("shaders/shadow.vert.spv", logicalDevice);
    VkShaderModule fragShaderModule =
        ShaderModule::Create("shaders/shadow.frag.spv", logicalDevice);

    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {};
    shaderStages[0].sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shaderStages[0].module = vertShaderModule;
    shaderStages[0].pName = "main";
    shaderStages[1].sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStages[1].module = fragShaderModule;
    shaderStages[1].pName = "main";

    const VkVertexInputBindingDescription bindingDescription =
        Vertex::getBindingDescription();
    const std::array<VkVertexInputAttributeDescription, 5>
        attributeDescriptions = Vertex::getAttributeDescriptions();
    VkPipelineVertexInputStateCreateInfo vertexInput = {};
    vertexInput.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &bindingDescription;
    vertexInput.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(attributeDescriptions.size());
    vertexInput.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport = {};
    viewport.width = static_cast<float>(ShadowMapSize);
    viewport.height = static_cast<float>(ShadowMapSize);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    VkRect2D scissor = {};
    scissor.extent = { ShadowMapSize, ShadowMapSize };
    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;
    rasterizer.depthBiasEnable = VK_TRUE;
    rasterizer.depthBiasConstantFactor = 1.25f;
    rasterizer.depthBiasSlopeFactor = 1.75f;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 0;

    const std::array<VkDescriptorSetLayout, 2> descriptorSetLayouts = {
        shadowDescriptorSetLayout, modelDescriptorSetLayout
    };
    VkPipelineLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount =
        static_cast<uint32_t>(descriptorSetLayouts.size());
    layoutInfo.pSetLayouts = descriptorSetLayouts.data();
    if (vkCreatePipelineLayout(logicalDevice, &layoutInfo, nullptr,
                               &shadowPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shadow pipeline layout");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = shadowPipelineLayout;
    pipelineInfo.renderPass = shadowRenderPass;
    pipelineInfo.subpass = 0;
    // Opaque casters only need the vertex shader and fixed-function depth
    // output. Alpha mask/blend casters need the fragment shader to sample the
    // base-color alpha and discard transparent texels.
    pipelineInfo.stageCount = 1;
    pipelineInfo.pStages = shaderStages.data();
    if (vkCreateGraphicsPipelines(logicalDevice, VK_NULL_HANDLE, 1,
                                  &pipelineInfo, nullptr,
                                  &shadowOpaquePipeline) != VK_SUCCESS) {
        throw std::runtime_error(
            "Failed to create opaque shadow pipeline");
    }
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    if (vkCreateGraphicsPipelines(logicalDevice, VK_NULL_HANDLE, 1,
                                  &pipelineInfo, nullptr,
                                  &shadowAlphaPipeline) != VK_SUCCESS) {
        throw std::runtime_error(
            "Failed to create alpha shadow pipeline");
    }

    vkDestroyShaderModule(logicalDevice, vertShaderModule, nullptr);
    vkDestroyShaderModule(logicalDevice, fragShaderModule, nullptr);
}

void Renderer::CreateGrassPipeline() {
    // --- Set up programmable shaders ---
    VkShaderModule vertShaderModule = ShaderModule::Create("shaders/grass.vert.spv", logicalDevice);
    VkShaderModule tescShaderModule = ShaderModule::Create("shaders/grass.tesc.spv", logicalDevice);
    VkShaderModule teseShaderModule = ShaderModule::Create("shaders/grass.tese.spv", logicalDevice);
    VkShaderModule fragShaderModule = ShaderModule::Create("shaders/grass.frag.spv", logicalDevice);

    // Assign each shader module to the appropriate stage in the pipeline
    VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo tescShaderStageInfo = {};
    tescShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    tescShaderStageInfo.stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
    tescShaderStageInfo.module = tescShaderModule;
    tescShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo teseShaderStageInfo = {};
    teseShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    teseShaderStageInfo.stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    teseShaderStageInfo.module = teseShaderModule;
    teseShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, tescShaderStageInfo, teseShaderStageInfo, fragShaderStageInfo };

    // --- Set up fixed-function stages ---

    // Vertex input
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    auto bindingDescription = Blade::getBindingDescription();
    auto attributeDescriptions = Blade::getAttributeDescriptions();

    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    // Input Assembly
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // Viewports and Scissors (rectangles that define in which regions pixels are stored)
    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(swapChain->GetVkExtent().width);
    viewport.height = static_cast<float>(swapChain->GetVkExtent().height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = { 0, 0 };
    scissor.extent = swapChain->GetVkExtent();

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    // Rasterizer
    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f;
    rasterizer.depthBiasClamp = 0.0f;
    rasterizer.depthBiasSlopeFactor = 0.0f;

    // Multisampling (turned off here)
    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f;
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;

    // Depth testing
    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f;
    depthStencil.maxDepthBounds = 1.0f;
    depthStencil.stencilTestEnable = VK_FALSE;

    // Color blending (turned off here, but showing options for learning)
    // --> Configuration per attached framebuffer
    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    // --> Global color blending settings
    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    std::vector<VkDescriptorSetLayout> descriptorSetLayouts = {
        cameraDescriptorSetLayout,
        modelDescriptorSetLayout,
        shadowDescriptorSetLayout
    };

    // Pipeline layout: used to specify uniform values
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
    pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = 0;

    if (vkCreatePipelineLayout(logicalDevice, &pipelineLayoutInfo, nullptr, &grassPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout");
    }

    // Tessellation state
    VkPipelineTessellationStateCreateInfo tessellationInfo = {};
    tessellationInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
    tessellationInfo.pNext = NULL;
    tessellationInfo.flags = 0;
    tessellationInfo.patchControlPoints = 1;

    // --- Create graphics pipeline ---
    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 4;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pTessellationState = &tessellationInfo;
    pipelineInfo.pDynamicState = nullptr;
    pipelineInfo.layout = grassPipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    if (vkCreateGraphicsPipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &grassPipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create graphics pipeline");
    }

    // No need for the shader modules anymore
    vkDestroyShaderModule(logicalDevice, vertShaderModule, nullptr);
    vkDestroyShaderModule(logicalDevice, tescShaderModule, nullptr);
    vkDestroyShaderModule(logicalDevice, teseShaderModule, nullptr);
    vkDestroyShaderModule(logicalDevice, fragShaderModule, nullptr);
}

void Renderer::CreateFlowerPipeline() {
    VkShaderModule vertShaderModule =
        ShaderModule::Create("shaders/flower.vert.spv", logicalDevice);
    VkShaderModule fragShaderModule =
        ShaderModule::Create("shaders/flower.frag.spv", logicalDevice);

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertShaderModule;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragShaderModule;
    stages[1].pName = "main";

    std::array<VkVertexInputBindingDescription, 2> bindings = {};
    bindings[0] = FlowerVertex::getBindingDescription();
    bindings[1].binding = 1;
    bindings[1].stride = sizeof(Blade);
    bindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    std::array<VkVertexInputAttributeDescription, 7> attributes = {};
    const auto flowerAttributes = FlowerVertex::getAttributeDescriptions();
    for (std::size_t index = 0; index < flowerAttributes.size(); ++index) {
        attributes[index] = flowerAttributes[index];
    }
    const std::size_t bladeOffsets[4] = {
        offsetof(Blade, v0), offsetof(Blade, v1),
        offsetof(Blade, v2), offsetof(Blade, up)
    };
    for (std::uint32_t index = 0; index < 4; ++index) {
        attributes[3 + index].binding = 1;
        attributes[3 + index].location = 3 + index;
        attributes[3 + index].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributes[3 + index].offset =
            static_cast<std::uint32_t>(bladeOffsets[index]);
    }

    VkPipelineVertexInputStateCreateInfo vertexInput = {};
    vertexInput.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount =
        static_cast<std::uint32_t>(bindings.size());
    vertexInput.pVertexBindingDescriptions = bindings.data();
    vertexInput.vertexAttributeDescriptionCount =
        static_cast<std::uint32_t>(attributes.size());
    vertexInput.pVertexAttributeDescriptions = attributes.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport = {};
    viewport.width = static_cast<float>(swapChain->GetVkExtent().width);
    viewport.height = static_cast<float>(swapChain->GetVkExtent().height);
    viewport.maxDepth = 1.0f;
    VkRect2D scissor = {};
    scissor.extent = swapChain->GetVkExtent();
    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState blendAttachment = {};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blending = {};
    blending.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blending.attachmentCount = 1;
    blending.pAttachments = &blendAttachment;

    std::array<VkDescriptorSetLayout, 3> descriptorLayouts = {
        cameraDescriptorSetLayout,
        modelDescriptorSetLayout,
        shadowDescriptorSetLayout
    };
    VkPushConstantRange pushRange = {};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushRange.size = sizeof(FlowerRenderPushConstants);
    VkPipelineLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount =
        static_cast<std::uint32_t>(descriptorLayouts.size());
    layoutInfo.pSetLayouts = descriptorLayouts.data();
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(logicalDevice, &layoutInfo, nullptr,
                               &flowerPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create flower pipeline layout");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &blending;
    pipelineInfo.layout = flowerPipelineLayout;
    pipelineInfo.renderPass = renderPass;
    if (vkCreateGraphicsPipelines(logicalDevice, VK_NULL_HANDLE, 1,
                                  &pipelineInfo, nullptr,
                                  &flowerPipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create flower pipeline");
    }

    vkDestroyShaderModule(logicalDevice, vertShaderModule, nullptr);
    vkDestroyShaderModule(logicalDevice, fragShaderModule, nullptr);
}

void Renderer::CreateComputePipeline() {
    // Set up programmable shaders
    VkShaderModule computeShaderModule = ShaderModule::Create("shaders/compute.comp.spv", logicalDevice);

    VkPipelineShaderStageCreateInfo computeShaderStageInfo = {};
    computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computeShaderStageInfo.module = computeShaderModule;
    computeShaderStageInfo.pName = "main";

    // TODO: Add the compute dsecriptor set layout you create to this list
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts = { cameraDescriptorSetLayout, timeDescriptorSetLayout, computeDescriptorSetLayout };

    // Create pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
    pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
    VkPushConstantRange flowerCullRange = {};
    flowerCullRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    flowerCullRange.offset = 0;
    flowerCullRange.size = sizeof(FlowerCullPushConstants);
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &flowerCullRange;

    if (vkCreatePipelineLayout(logicalDevice, &pipelineLayoutInfo, nullptr, &computePipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout");
    }

    // Create compute pipeline
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = computeShaderStageInfo;
    pipelineInfo.layout = computePipelineLayout;
    pipelineInfo.pNext = nullptr;
    pipelineInfo.flags = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    if (vkCreateComputePipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipeline");
    }

    // No need for shader modules anymore
    vkDestroyShaderModule(logicalDevice, computeShaderModule, nullptr);
}

void Renderer::CreateFrameResources() {
    imageViews.resize(swapChain->GetCount());

    for (uint32_t i = 0; i < swapChain->GetCount(); i++) {
        // --- Create an image view for each swap chain image ---
        VkImageViewCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = swapChain->GetVkImage(i);

        // Specify how the image data should be interpreted
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = swapChain->GetVkImageFormat();

        // Specify color channel mappings (can be used for swizzling)
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

        // Describe the image's purpose and which part of the image should be accessed
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        // Create the image view
        if (vkCreateImageView(logicalDevice, &createInfo, nullptr, &imageViews[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image views");
        }
    }

    VkFormat depthFormat = device->GetInstance()->GetSupportedFormat({ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT }, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    // CREATE DEPTH IMAGE
    Image::Create(device,
        swapChain->GetVkExtent().width,
        swapChain->GetVkExtent().height,
        depthFormat,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        depthImage,
        depthImageMemory
    );

    depthImageView = Image::CreateView(device, depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
    
    // Transition the image for use as depth-stencil
    Image::TransitionLayout(device, graphicsCommandPool, depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    
    // CREATE FRAMEBUFFERS
    framebuffers.resize(swapChain->GetCount());
    for (size_t i = 0; i < swapChain->GetCount(); i++) {
        std::vector<VkImageView> attachments = {
            imageViews[i],
            depthImageView
        };

        VkFramebufferCreateInfo framebufferInfo = {};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = swapChain->GetVkExtent().width;
        framebufferInfo.height = swapChain->GetVkExtent().height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(logicalDevice, &framebufferInfo, nullptr, &framebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create framebuffer");
        }

    }

    CreateTimestampQueryResources();
}

void Renderer::DestroyFrameResources() {
    DestroyTimestampQueryResources();

    for (size_t i = 0; i < imageViews.size(); i++) {
        vkDestroyImageView(logicalDevice, imageViews[i], nullptr);
    }

    vkDestroyImageView(logicalDevice, depthImageView, nullptr);
    vkFreeMemory(logicalDevice, depthImageMemory, nullptr);
    vkDestroyImage(logicalDevice, depthImage, nullptr);

    for (size_t i = 0; i < framebuffers.size(); i++) {
        vkDestroyFramebuffer(logicalDevice, framebuffers[i], nullptr);
    }
}

void Renderer::CreateTimestampQueryResources() {
    gpuProfilerSupported = false;
    hasGpuProfilerResults = false;
    timestampValidBits = 0;
    timestampPeriodNanoseconds = 0.0;
    graphicsGpuTimeMs = 0.0;
    shadowGpuTimeMs = 0.0;
    mainGpuTimeMs = 0.0;

    VkPhysicalDeviceProperties deviceProperties = {};
    vkGetPhysicalDeviceProperties(
        device->GetInstance()->GetPhysicalDevice(), &deviceProperties);

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(
        device->GetInstance()->GetPhysicalDevice(), &queueFamilyCount,
        nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(
        device->GetInstance()->GetPhysicalDevice(), &queueFamilyCount,
        queueFamilies.data());

    const uint32_t graphicsQueueFamily =
        device->GetQueueIndex(QueueFlags::Graphics);
    if (graphicsQueueFamily >= queueFamilies.size() ||
        queueFamilies[graphicsQueueFamily].timestampValidBits == 0 ||
        deviceProperties.limits.timestampPeriod <= 0.0f) {
        std::cout << "GPU timestamp profiler unavailable on the graphics "
                     "queue.\n";
        return;
    }

    timestampValidBits =
        queueFamilies[graphicsQueueFamily].timestampValidBits;
    timestampPeriodNanoseconds =
        static_cast<double>(deviceProperties.limits.timestampPeriod);
    timestampQueryPools.assign(swapChain->GetCount(), VK_NULL_HANDLE);
    timestampQueriesSubmitted.assign(swapChain->GetCount(), false);

    VkQueryPoolCreateInfo queryPoolInfo = {};
    queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolInfo.queryCount = TimestampQueryCount;

    for (std::size_t index = 0; index < timestampQueryPools.size(); ++index) {
        if (vkCreateQueryPool(logicalDevice, &queryPoolInfo, nullptr,
                              &timestampQueryPools[index]) != VK_SUCCESS) {
            DestroyTimestampQueryResources();
            throw std::runtime_error(
                "Failed to create GPU timestamp query pool");
        }
    }

    gpuProfilerSupported = true;
    std::cout << "GPU timestamp profiler enabled: period="
              << timestampPeriodNanoseconds << "ns, validBits="
              << timestampValidBits << ".\n";
}

void Renderer::DestroyTimestampQueryResources() {
    for (std::size_t index = 0; index < timestampQueryPools.size(); ++index) {
        if (timestampQueryPools[index] != VK_NULL_HANDLE) {
            vkDestroyQueryPool(logicalDevice, timestampQueryPools[index],
                               nullptr);
        }
    }
    timestampQueryPools.clear();
    timestampQueriesSubmitted.clear();
    gpuProfilerSupported = false;
    hasGpuProfilerResults = false;
}

double Renderer::TimestampDeltaMilliseconds(uint64_t begin,
                                            uint64_t end) const {
    uint64_t delta = end - begin;
    if (timestampValidBits < 64) {
        const uint64_t mask = (uint64_t(1) << timestampValidBits) - 1;
        delta &= mask;
    }
    return static_cast<double>(delta) * timestampPeriodNanoseconds /
           1000000.0;
}

void Renderer::ReadTimestampResults(uint32_t imageIndex) {
    if (!gpuProfilerSupported ||
        imageIndex >= timestampQueryPools.size() ||
        !timestampQueriesSubmitted[imageIndex]) {
        return;
    }

    // Each query returns { timestamp, availability }. The corresponding
    // swapchain-image fence has already completed, so this read never needs
    // VK_QUERY_RESULT_WAIT_BIT and cannot introduce a profiler stall.
    std::array<uint64_t, TimestampQueryCount * 2> queryData = {};
    const VkResult result = vkGetQueryPoolResults(
        logicalDevice, timestampQueryPools[imageIndex], 0,
        TimestampQueryCount, sizeof(queryData), queryData.data(),
        sizeof(uint64_t) * 2,
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (result != VK_SUCCESS) {
        return;
    }
    for (uint32_t query = 0; query < TimestampQueryCount; ++query) {
        if (queryData[query * 2 + 1] == 0) {
            return;
        }
    }

    const auto timestamp = [&](TimestampQuery query) {
        return queryData[static_cast<uint32_t>(query) * 2];
    };
    const double graphicsSample = TimestampDeltaMilliseconds(
        timestamp(TimestampGraphicsBegin),
        timestamp(TimestampGraphicsEnd));
    const double shadowSample = TimestampDeltaMilliseconds(
        timestamp(TimestampShadowBegin), timestamp(TimestampShadowEnd));
    const double mainSample = TimestampDeltaMilliseconds(
        timestamp(TimestampMainBegin), timestamp(TimestampMainEnd));

    if (!hasGpuProfilerResults) {
        graphicsGpuTimeMs = graphicsSample;
        shadowGpuTimeMs = shadowSample;
        mainGpuTimeMs = mainSample;
    } else {
        // A short exponential moving average keeps the title readable while
        // still responding quickly when the viewed part of the scene changes.
        constexpr double smoothing = 0.10;
        graphicsGpuTimeMs +=
            (graphicsSample - graphicsGpuTimeMs) * smoothing;
        shadowGpuTimeMs += (shadowSample - shadowGpuTimeMs) * smoothing;
        mainGpuTimeMs += (mainSample - mainGpuTimeMs) * smoothing;
    }
    hasGpuProfilerResults = true;
}

void Renderer::RecreateFrameResources() {
    vkDeviceWaitIdle(logicalDevice);

    vkDestroyPipeline(logicalDevice, graphicsOpaquePipeline, nullptr);
    vkDestroyPipeline(logicalDevice, graphicsOpaqueDoubleSidedPipeline,
                      nullptr);
    vkDestroyPipeline(logicalDevice, graphicsMaskPipeline, nullptr);
    vkDestroyPipeline(logicalDevice, graphicsMaskDoubleSidedPipeline,
                      nullptr);
    vkDestroyPipeline(logicalDevice, graphicsBlendPipeline, nullptr);
    vkDestroyPipeline(logicalDevice, graphicsBlendDoubleSidedPipeline,
                      nullptr);
    vkDestroyPipeline(logicalDevice, grassPipeline, nullptr);
    vkDestroyPipeline(logicalDevice, flowerPipeline, nullptr);
    vkDestroyPipeline(logicalDevice, skyboxPipeline, nullptr);

    vkDestroyPipelineLayout(logicalDevice, graphicsPipelineLayout, nullptr);
    vkDestroyPipelineLayout(logicalDevice, grassPipelineLayout, nullptr);
    vkDestroyPipelineLayout(logicalDevice, flowerPipelineLayout, nullptr);
    vkDestroyPipelineLayout(logicalDevice, skyboxPipelineLayout, nullptr);

    DestroyCommandBuffers();

    DestroyFrameResources();
    CreateFrameResources();
    CreateGraphicsPipeline();
    CreateGrassPipeline();
    CreateFlowerPipeline();
    CreateSkyboxPipeline();
    RecordCommandBuffers();
}

void Renderer::RecordComputeCommandBuffers() {
    const uint32_t frameCount = swapChain->GetCount();
    computeCommandBuffers.resize(frameCount);
    computeFinishedSemaphores.resize(frameCount, VK_NULL_HANDLE);

    // Specify the command pool and number of buffers to allocate
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = computeCommandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = frameCount;

    if (vkAllocateCommandBuffers(logicalDevice, &allocInfo,
                                 computeCommandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffers");
    }

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
        if (vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr,
                              &computeFinishedSemaphores[frameIndex]) !=
            VK_SUCCESS) {
            throw std::runtime_error(
                "Failed to create compute-finished semaphore");
        }

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        if (vkBeginCommandBuffer(computeCommandBuffers[frameIndex],
                                 &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error(
                "Failed to begin recording compute command buffer");
        }

        VkCommandBuffer commandBuffer = computeCommandBuffers[frameIndex];
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          computePipeline);
        vkCmdBindDescriptorSets(
            commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            computePipelineLayout, 0, 1,
            &cameraDescriptorSets[frameIndex], 0, nullptr);
        vkCmdBindDescriptorSets(
            commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            computePipelineLayout, 1, 1, &timeDescriptorSet, 0, nullptr);

        for (uint32_t j = 0; j < scene->GetBlades().size(); ++j) {
            Blades* blades = scene->GetBlades()[j];

        // Clear the indirect vertex count before culling. A shader barrier is
        // only workgroup-local, so resetting it inside compute.comp races as
        // soon as a field contains more than one workgroup.
            vkCmdFillBuffer(commandBuffer,
                        blades->GetNumBladesBuffer(), 0,
                        sizeof(std::uint32_t), 0);
            vkCmdFillBuffer(commandBuffer,
                            blades->GetFlowerDrawIndirectBuffer(),
                            offsetof(VkDrawIndexedIndirectCommand,
                                     instanceCount),
                            sizeof(std::uint32_t), 0);
        std::array<VkBufferMemoryBarrier, 2> resetBarriers = {};
        resetBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        resetBarriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        resetBarriers[0].dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        resetBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        resetBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        resetBarriers[0].buffer = blades->GetNumBladesBuffer();
        resetBarriers[0].offset = 0;
        resetBarriers[0].size = sizeof(std::uint32_t);
        resetBarriers[1] = resetBarriers[0];
        resetBarriers[1].buffer = blades->GetFlowerDrawIndirectBuffer();
        resetBarriers[1].offset = offsetof(
            VkDrawIndexedIndirectCommand, instanceCount);
        resetBarriers[1].size = sizeof(std::uint32_t);
            vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
            0, nullptr, static_cast<std::uint32_t>(resetBarriers.size()),
            resetBarriers.data(), 0, nullptr);

        // Bind compute descriptor set
            vkCmdBindDescriptorSets(commandBuffer,
                VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 2, 1,
                &computeDescriptorSets[j], 0, nullptr);

            FlowerCullPushConstants flowerCull = {};
            flowerCull.density = blades->GetFlowerDensity();
            flowerCull.enabled = blades->GetFlowersEnabled() ? 1u : 0u;
            vkCmdPushConstants(
                commandBuffer, computePipelineLayout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0,
                sizeof(flowerCull), &flowerCull);

        // Dispatch compute shader
        uint32_t numWorkGroups =
            (blades->GetBladeCount() + WORKGROUP_SIZE - 1) /
            WORKGROUP_SIZE;
            vkCmdDispatch(commandBuffer, numWorkGroups, 1, 1);
        }

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error(
                "Failed to record compute command buffer");
        }
    }
}

void Renderer::RecordCommandBuffers() {
    commandBuffers.resize(swapChain->GetCount());

    // Specify the command pool and number of buffers to allocate
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = graphicsCommandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

    if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffers");
    }

    commandBufferFences.assign(commandBuffers.size(), VK_NULL_HANDLE);
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (std::size_t index = 0; index < commandBufferFences.size(); ++index) {
        if (vkCreateFence(logicalDevice, &fenceInfo, nullptr,
                          &commandBufferFences[index]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create command buffer fence");
        }
    }
}

void Renderer::DestroyCommandBuffers() {
    for (std::size_t index = 0; index < commandBufferFences.size(); ++index) {
        if (commandBufferFences[index] != VK_NULL_HANDLE) {
            vkDestroyFence(logicalDevice, commandBufferFences[index], nullptr);
        }
    }
    commandBufferFences.clear();
    if (!commandBuffers.empty()) {
        vkFreeCommandBuffers(
            logicalDevice, graphicsCommandPool,
            static_cast<uint32_t>(commandBuffers.size()),
            commandBuffers.data());
        commandBuffers.clear();
    }
}

void Renderer::UpdateVisibleModels() {
    const std::vector<Model*>& models = scene->GetModels();
    visibleModelIndices.clear();
    visibleModelIndices.reserve(models.size());
    visibleOpaqueSingleSidedIndices.clear();
    visibleOpaqueDoubleSidedIndices.clear();
    visibleMaskSingleSidedIndices.clear();
    visibleMaskDoubleSidedIndices.clear();
    visibleBlendIndices.clear();
    shadowCasterIndices.clear();
    shadowCasterIndices.reserve(models.size());
    shadowOpaqueCasterIndices.clear();
    shadowAlphaCasterIndices.clear();
    shadowSubpixelCulledCount = 0;
    shadowReceiverCulledCount = 0;

    if (!frustumCullingEnabled) {
        for (uint32_t index = 0;
             index < static_cast<uint32_t>(models.size()); ++index) {
            visibleModelIndices.push_back(index);
        }
    } else {
        const Frustum frustum(
            camera->GetProjectionMatrix() * camera->GetViewMatrix());
        for (uint32_t index = 0;
             index < static_cast<uint32_t>(models.size()); ++index) {
            const Model* model = models[index];
            if (!model->HasBounds() || frustum.IntersectsAabb(
                    model->GetBoundsMin(), model->GetBoundsMax())) {
                visibleModelIndices.push_back(index);
            }
        }
    }

    visibleModelCount =
        static_cast<uint32_t>(visibleModelIndices.size());
    culledModelCount = static_cast<uint32_t>(models.size()) -
                       visibleModelCount;

    if (!shadowCullingEnabled) {
        for (uint32_t index = 0;
             index < static_cast<uint32_t>(models.size()); ++index) {
            shadowCasterIndices.push_back(index);
        }
    } else {
        const Frustum shadowFrustum(
            shadowBufferObject.lightViewProjection);
        const Frustum receiverFrustum(
            camera->GetProjectionMatrix() * camera->GetViewMatrix());
        const glm::vec3 shadowDirection = -glm::normalize(
            glm::vec3(shadowBufferObject.lightDirection));
        // Sweep through the complete configured shadow depth. This keeps the
        // receiver test conservative: a caster is removed only when its
        // shadow cannot reach any point inside the camera frustum.
        constexpr float maximumShadowDistance = 40.0f;
        const glm::vec3 shadowOffset =
            shadowDirection * maximumShadowDistance;
        for (uint32_t index = 0;
             index < static_cast<uint32_t>(models.size()); ++index) {
            const Model* model = models[index];
            if (!model->HasBounds() || shadowFrustum.IntersectsAabb(
                    model->GetBoundsMin(), model->GetBoundsMax())) {
                if (model->HasBounds()) {
                    const glm::vec3 sweptMin = glm::min(
                        model->GetBoundsMin(),
                        model->GetBoundsMin() + shadowOffset);
                    const glm::vec3 sweptMax = glm::max(
                        model->GetBoundsMax(),
                        model->GetBoundsMax() + shadowOffset);
                    if (!receiverFrustum.IntersectsAabb(
                            sweptMin, sweptMax)) {
                        ++shadowReceiverCulledCount;
                        continue;
                    }
                }
                if (model->HasBounds() &&
                    ProjectedShadowExtentPixels(
                        model->GetBoundsMin(), model->GetBoundsMax(),
                        shadowBufferObject.lightViewProjection,
                        ShadowMapSize) < 1.0f) {
                    ++shadowSubpixelCulledCount;
                    continue;
                }
                shadowCasterIndices.push_back(index);
            }
        }
    }
    shadowCasterCount =
        static_cast<uint32_t>(shadowCasterIndices.size());
    shadowCulledModelCount = static_cast<uint32_t>(models.size()) -
                             shadowCasterCount;
    for (std::size_t shadowIndex = 0;
         shadowIndex < shadowCasterIndices.size(); ++shadowIndex) {
        const uint32_t modelIndex = shadowCasterIndices[shadowIndex];
        if (models[modelIndex]->GetAlphaMode() ==
            MaterialAlphaMode::Opaque) {
            shadowOpaqueCasterIndices.push_back(modelIndex);
        } else {
            shadowAlphaCasterIndices.push_back(modelIndex);
        }
    }

    for (std::size_t visibleIndex = 0;
         visibleIndex < visibleModelIndices.size(); ++visibleIndex) {
        const uint32_t modelIndex = visibleModelIndices[visibleIndex];
        const Model* model = models[modelIndex];
        switch (model->GetAlphaMode()) {
        case MaterialAlphaMode::Opaque:
            (model->IsDoubleSided()
                 ? visibleOpaqueDoubleSidedIndices
                 : visibleOpaqueSingleSidedIndices).push_back(modelIndex);
            break;
        case MaterialAlphaMode::Mask:
            (model->IsDoubleSided()
                 ? visibleMaskDoubleSidedIndices
                 : visibleMaskSingleSidedIndices).push_back(modelIndex);
            break;
        case MaterialAlphaMode::Blend:
            visibleBlendIndices.push_back(modelIndex);
            break;
        }
    }

    // True alpha blending is order-dependent. Sorting whole primitives is the
    // best ordering available without per-triangle sorting or OIT.
    const glm::vec3 cameraPosition =
        glm::vec3(glm::inverse(camera->GetViewMatrix())[3]);
    std::sort(visibleBlendIndices.begin(), visibleBlendIndices.end(),
        [&](uint32_t left, uint32_t right) {
            const glm::vec3 leftCenter =
                (models[left]->GetBoundsMin() +
                 models[left]->GetBoundsMax()) * 0.5f;
            const glm::vec3 rightCenter =
                (models[right]->GetBoundsMin() +
                 models[right]->GetBoundsMax()) * 0.5f;
            return glm::dot(leftCenter - cameraPosition,
                            leftCenter - cameraPosition) >
                   glm::dot(rightCenter - cameraPosition,
                            rightCenter - cameraPosition);
        });
    if (!hasReportedCullingStats) {
        std::cout << "FRUSTUM_CULLING enabled="
                  << (frustumCullingEnabled ? 1 : 0)
                  << " visible=" << visibleModelCount
                  << " culled=" << culledModelCount
                  << " total=" << models.size()
                  << " queues={opaque:"
                  << (visibleOpaqueSingleSidedIndices.size() +
                      visibleOpaqueDoubleSidedIndices.size())
                  << ",mask:"
                  << (visibleMaskSingleSidedIndices.size() +
                      visibleMaskDoubleSidedIndices.size())
                  << ",blend:" << visibleBlendIndices.size()
                  << ",doubleSided:"
                  << (visibleOpaqueDoubleSidedIndices.size() +
                      visibleMaskDoubleSidedIndices.size())
                  << "} shadow={enabled:"
                  << (shadowCullingEnabled ? 1 : 0)
                  << ",casters:" << shadowCasterCount
                  << ",culled:" << shadowCulledModelCount
                  << ",subpixel:" << shadowSubpixelCulledCount
                  << ",receiver:" << shadowReceiverCulledCount
                  << ",opaque:" << shadowOpaqueCasterIndices.size()
                  << ",alpha:" << shadowAlphaCasterIndices.size()
                  << "}\n";
        hasReportedCullingStats = true;
    }
}

void Renderer::RecordCommandBuffer(uint32_t imageIndex) {
        const std::size_t i = imageIndex;
        UpdateVisibleModels();
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        beginInfo.pInheritanceInfo = nullptr;

        // ~ Start recording ~
        if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("Failed to begin recording command buffer");
        }

        const bool profileGpu = gpuProfilerSupported &&
            i < timestampQueryPools.size();
        if (profileGpu) {
            vkCmdResetQueryPool(commandBuffers[i], timestampQueryPools[i], 0,
                                TimestampQueryCount);
            vkCmdWriteTimestamp(commandBuffers[i],
                                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                timestampQueryPools[i],
                                TimestampGraphicsBegin);
            vkCmdWriteTimestamp(commandBuffers[i],
                                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                timestampQueryPools[i],
                                TimestampShadowBegin);
        }

        // Begin the render pass
        VkRenderPassBeginInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[i];
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChain->GetVkExtent();

        std::array<VkClearValue, 2> clearValues = {};
        clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
        clearValues[1].depthStencil = { 1.0f, 0 };
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        std::vector<VkBufferMemoryBarrier> barriers(
            scene->GetBlades().size() * 3);
        for (uint32_t j = 0; j < scene->GetBlades().size(); ++j) {
            const uint32_t base = 3 * j;
            barriers[base].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barriers[base].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barriers[base].dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
            barriers[base].srcQueueFamilyIndex = device->GetQueueIndex(QueueFlags::Compute);
            barriers[base].dstQueueFamilyIndex = device->GetQueueIndex(QueueFlags::Graphics);
            barriers[base].buffer = scene->GetBlades()[j]->GetNumBladesBuffer();
            barriers[base].offset = 0;
            barriers[base].size = sizeof(BladeDrawIndirect);
            barriers[base + 1] = barriers[base];
            barriers[base + 1].buffer =
                scene->GetBlades()[j]->GetFlowerDrawIndirectBuffer();
            barriers[base + 1].size = sizeof(VkDrawIndexedIndirectCommand);
            barriers[base + 2] = barriers[base];
            barriers[base + 2].dstAccessMask =
                VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
            barriers[base + 2].buffer =
                scene->GetBlades()[j]->GetCulledFlowersBuffer();
            barriers[base + 2].size = VK_WHOLE_SIZE;
        }

        vkCmdPipelineBarrier(
            commandBuffers[i], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
                VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
            0, 0, nullptr,
            static_cast<uint32_t>(barriers.size()), barriers.data(), 0,
            nullptr);

        // Render all ordinary models from the directional light before the
        // camera pass. The shadow render pass leaves the image shader-readable.
        VkClearValue shadowClear = {};
        shadowClear.depthStencil = { 1.0f, 0 };
        VkRenderPassBeginInfo shadowPassInfo = {};
        shadowPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        shadowPassInfo.renderPass = shadowRenderPass;
        shadowPassInfo.framebuffer = shadowFramebuffer;
        shadowPassInfo.renderArea.offset = { 0, 0 };
        shadowPassInfo.renderArea.extent = { ShadowMapSize, ShadowMapSize };
        shadowPassInfo.clearValueCount = 1;
        shadowPassInfo.pClearValues = &shadowClear;

        vkCmdBeginRenderPass(commandBuffers[i], &shadowPassInfo,
                             VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindDescriptorSets(
            commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
            shadowPipelineLayout, 0, 1, &shadowDescriptorSet, 0, nullptr);
        const auto drawShadowQueue = [&](
            VkPipeline pipeline,
            const std::vector<uint32_t>& casterIndices) {
            if (casterIndices.empty()) {
                return;
            }
            vkCmdBindPipeline(commandBuffers[i],
                              VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            for (std::size_t shadowIndex = 0;
                 shadowIndex < casterIndices.size(); ++shadowIndex) {
                const uint32_t j = casterIndices[shadowIndex];
                const VkBuffer vertexBuffers[] = {
                    scene->GetModels()[j]->getVertexBuffer()
                };
                const VkDeviceSize offsets[] = { 0 };
                vkCmdBindVertexBuffers(commandBuffers[i], 0, 1,
                                       vertexBuffers, offsets);
                vkCmdBindIndexBuffer(
                    commandBuffers[i],
                    scene->GetModels()[j]->getIndexBuffer(), 0,
                    VK_INDEX_TYPE_UINT32);
                vkCmdBindDescriptorSets(
                    commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                    shadowPipelineLayout, 1, 1, &modelDescriptorSets[j], 0,
                    nullptr);
                vkCmdDrawIndexed(
                    commandBuffers[i],
                    static_cast<uint32_t>(
                        scene->GetModels()[j]->getIndices().size()),
                    1, 0, 0, 0);
            }
        };
        drawShadowQueue(shadowOpaquePipeline,
                        shadowOpaqueCasterIndices);
        drawShadowQueue(shadowAlphaPipeline,
                        shadowAlphaCasterIndices);
        vkCmdEndRenderPass(commandBuffers[i]);
        if (profileGpu) {
            vkCmdWriteTimestamp(commandBuffers[i],
                                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                timestampQueryPools[i], TimestampShadowEnd);
            vkCmdWriteTimestamp(commandBuffers[i],
                                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                timestampQueryPools[i], TimestampMainBegin);
        }

        // Each swapchain image owns an independent camera UBO, so a newly
        // acquired frame never observes a buffer still used by another frame.
        vkCmdBindDescriptorSets(commandBuffers[i],
            VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 0, 1,
            &cameraDescriptorSets[i], 0, nullptr);

        vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		// Bind the skybox pipeline
        vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, skyboxPipeline);
        vkCmdBindDescriptorSets(commandBuffers[i],
            VK_PIPELINE_BIND_POINT_GRAPHICS, skyboxPipelineLayout, 0, 1,
            &cameraDescriptorSets[i], 0, nullptr);
        vkCmdBindDescriptorSets(
            commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
            skyboxPipelineLayout, 1, 1, &environmentDescriptorSet, 0,
            nullptr);
        VkBuffer skyboxBuffers[] = { skyboxVertexBuffer };
        VkDeviceSize skyboxOffsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, skyboxBuffers, skyboxOffsets);
        vkCmdDraw(commandBuffers[i], 36, 1, 0, 0);

        const auto bindMaterialPipeline = [&](VkPipeline pipeline) {
            vkCmdBindPipeline(commandBuffers[i],
                              VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            vkCmdBindDescriptorSets(
                commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                graphicsPipelineLayout, 0, 1, &cameraDescriptorSets[i], 0,
                nullptr);
            vkCmdBindDescriptorSets(
                commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                graphicsPipelineLayout, 2, 1, &shadowDescriptorSet, 0,
                nullptr);
            vkCmdBindDescriptorSets(
                commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                graphicsPipelineLayout, 3, 1, &environmentDescriptorSet, 0,
                nullptr);
        };
        const auto drawModel = [&](uint32_t modelIndex) {
            Model* model = scene->GetModels()[modelIndex];
            const VkBuffer vertexBuffers[] = { model->getVertexBuffer() };
            const VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers,
                                   offsets);
            vkCmdBindIndexBuffer(commandBuffers[i], model->getIndexBuffer(),
                                 0, VK_INDEX_TYPE_UINT32);
            vkCmdBindDescriptorSets(
                commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                graphicsPipelineLayout, 1, 1,
                &modelDescriptorSets[modelIndex], 0, nullptr);
            vkCmdDrawIndexed(
                commandBuffers[i],
                static_cast<uint32_t>(model->getIndices().size()),
                1, 0, 0, 0);
        };
        const auto drawQueue = [&](VkPipeline pipeline,
                                   const std::vector<uint32_t>& indices) {
            if (indices.empty()) {
                return;
            }
            bindMaterialPipeline(pipeline);
            for (std::size_t index = 0; index < indices.size(); ++index) {
                drawModel(indices[index]);
            }
        };

        drawQueue(graphicsOpaquePipeline,
                  visibleOpaqueSingleSidedIndices);
        drawQueue(graphicsOpaqueDoubleSidedPipeline,
                  visibleOpaqueDoubleSidedIndices);
        drawQueue(graphicsMaskPipeline, visibleMaskSingleSidedIndices);
        drawQueue(graphicsMaskDoubleSidedPipeline,
                  visibleMaskDoubleSidedIndices);

        // Blend primitives are globally sorted, so switch between the
        // single- and double-sided blend pipelines only when required.
        VkPipeline activeBlendPipeline = VK_NULL_HANDLE;
        for (std::size_t blendIndex = 0;
             blendIndex < visibleBlendIndices.size(); ++blendIndex) {
            const uint32_t modelIndex = visibleBlendIndices[blendIndex];
            const VkPipeline requiredPipeline =
                scene->GetModels()[modelIndex]->IsDoubleSided()
                    ? graphicsBlendDoubleSidedPipeline
                    : graphicsBlendPipeline;
            if (requiredPipeline != activeBlendPipeline) {
                bindMaterialPipeline(requiredPipeline);
                activeBlendPipeline = requiredPipeline;
            }
            drawModel(modelIndex);
        }

        // Bind the grass pipeline
        vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, grassPipeline);
        vkCmdBindDescriptorSets(
            commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
            grassPipelineLayout, 2, 1, &shadowDescriptorSet, 0, nullptr);

        for (uint32_t j = 0; j < scene->GetBlades().size(); ++j) {
            VkBuffer vertexBuffers[] = { scene->GetBlades()[j]->GetCulledBladesBuffer() };
            VkDeviceSize offsets[] = { 0 };
            // TODO: Uncomment this when the buffers are populated
            
            vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);

            // TODO: Bind the descriptor set for each grass blades model

            // Draw
            // TODO: Uncomment this when the buffers are populated
            // vkCmdDrawIndirect(commandBuffers[i], scene->GetBlades()[j]->GetNumBladesBuffer(), 0, 1, sizeof(BladeDrawIndirect));

            vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, grassPipelineLayout, 1, 1, &grassDescriptorSets[j], 0, nullptr);

            vkCmdDrawIndirect(commandBuffers[i], scene->GetBlades()[j]->GetNumBladesBuffer(), 0, 1, sizeof(BladeDrawIndirect));
        }

        // Flowers are ordinary low-poly instanced meshes. They deliberately
        // bypass the grass tessellation stages while consuming the compacted
        // instances produced by the same compute dispatch.
        vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                          flowerPipeline);
        vkCmdBindDescriptorSets(
            commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
            flowerPipelineLayout, 0, 1, &cameraDescriptorSets[i], 0,
            nullptr);
        vkCmdBindDescriptorSets(
            commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
            flowerPipelineLayout, 2, 1, &shadowDescriptorSet, 0, nullptr);
        for (uint32_t j = 0; j < scene->GetBlades().size(); ++j) {
            Blades* blades = scene->GetBlades()[j];
            if (!blades->GetFlowersEnabled()) {
                continue;
            }
            const VkBuffer vertexBuffers[] = {
                blades->GetFlowerVertexBuffer(),
                blades->GetCulledFlowersBuffer()
            };
            const VkDeviceSize offsets[] = { 0, 0 };
            vkCmdBindVertexBuffers(commandBuffers[i], 0, 2,
                                   vertexBuffers, offsets);
            vkCmdBindIndexBuffer(commandBuffers[i],
                                 blades->GetFlowerIndexBuffer(), 0,
                                 VK_INDEX_TYPE_UINT32);
            FlowerRenderPushConstants renderSettings = {};
            renderSettings.heightScale = blades->GetFlowerHeightScale();
            vkCmdPushConstants(
                commandBuffers[i], flowerPipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(renderSettings),
                &renderSettings);
            vkCmdDrawIndexedIndirect(
                commandBuffers[i], blades->GetFlowerDrawIndirectBuffer(),
                0, 1, sizeof(VkDrawIndexedIndirectCommand));
        }

        // End render pass
        vkCmdEndRenderPass(commandBuffers[i]);

        if (profileGpu) {
            vkCmdWriteTimestamp(commandBuffers[i],
                                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                timestampQueryPools[i], TimestampMainEnd);
            vkCmdWriteTimestamp(commandBuffers[i],
                                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                timestampQueryPools[i],
                                TimestampGraphicsEnd);
        }

        // ~ End recording ~
        if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to record command buffer");
        }
}

void Renderer::Frame() {
    if (!swapChain->Acquire()) {
        RecreateFrameResources();
        return;
    }

    const uint32_t imageIndex = swapChain->GetIndex();
    if (vkWaitForFences(logicalDevice, 1,
                        &commandBufferFences[imageIndex], VK_TRUE,
                        UINT64_MAX) != VK_SUCCESS) {
        throw std::runtime_error("Failed waiting for command buffer fence");
    }
    ReadTimestampResults(imageIndex);
    camera->Upload(imageIndex);

    VkSubmitInfo computeSubmitInfo = {};
    computeSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    computeSubmitInfo.commandBufferCount = 1;
    computeSubmitInfo.pCommandBuffers = &computeCommandBuffers[imageIndex];
    computeSubmitInfo.signalSemaphoreCount = 1;
    computeSubmitInfo.pSignalSemaphores =
        &computeFinishedSemaphores[imageIndex];
    if (vkQueueSubmit(device->GetQueue(QueueFlags::Compute), 1,
                      &computeSubmitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit compute command buffer");
    }

    if (vkResetFences(logicalDevice, 1,
                      &commandBufferFences[imageIndex]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to reset command buffer fence");
    }
    if (vkResetCommandBuffer(commandBuffers[imageIndex], 0) != VK_SUCCESS) {
        throw std::runtime_error("Failed to reset command buffer");
    }
    RecordCommandBuffer(imageIndex);

    // Submit the command buffer
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = {
        swapChain->GetImageAvailableVkSemaphore(),
        computeFinishedSemaphores[imageIndex]
    };
    VkPipelineStageFlags waitStages[] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
    };
    submitInfo.waitSemaphoreCount = 2;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

    VkSemaphore signalSemaphores[] = {
        swapChain->GetRenderFinishedVkSemaphore()
    };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(device->GetQueue(QueueFlags::Graphics), 1,
                      &submitInfo,
                      commandBufferFences[imageIndex]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit draw command buffer");
    }
    if (gpuProfilerSupported &&
        imageIndex < timestampQueriesSubmitted.size()) {
        timestampQueriesSubmitted[imageIndex] = true;
    }

    if (!swapChain->Present()) {
        RecreateFrameResources();
    }
}

uint32_t Renderer::GetVisibleModelCount() const {
    return visibleModelCount;
}

uint32_t Renderer::GetCulledModelCount() const {
    return culledModelCount;
}

uint32_t Renderer::GetTotalModelCount() const {
    return static_cast<uint32_t>(scene->GetModels().size());
}

bool Renderer::IsFrustumCullingEnabled() const {
    return frustumCullingEnabled;
}

uint32_t Renderer::GetShadowCasterCount() const {
    return shadowCasterCount;
}

uint32_t Renderer::GetShadowCulledModelCount() const {
    return shadowCulledModelCount;
}

bool Renderer::IsShadowCullingEnabled() const {
    return shadowCullingEnabled;
}

bool Renderer::IsGpuProfilerSupported() const {
    return gpuProfilerSupported;
}

bool Renderer::HasGpuProfilerResults() const {
    return hasGpuProfilerResults;
}

double Renderer::GetGraphicsGpuTimeMs() const {
    return graphicsGpuTimeMs;
}

double Renderer::GetShadowGpuTimeMs() const {
    return shadowGpuTimeMs;
}

double Renderer::GetMainGpuTimeMs() const {
    return mainGpuTimeMs;
}

Renderer::~Renderer() {
    vkDeviceWaitIdle(logicalDevice);

    // TODO: destroy any resources you created

    DestroyCommandBuffers();
    if (!computeCommandBuffers.empty()) {
        vkFreeCommandBuffers(
            logicalDevice, computeCommandPool,
            static_cast<uint32_t>(computeCommandBuffers.size()),
            computeCommandBuffers.data());
    }
    for (VkSemaphore semaphore : computeFinishedSemaphores) {
        if (semaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(logicalDevice, semaphore, nullptr);
        }
    }
    
    vkDestroyPipeline(logicalDevice, graphicsOpaquePipeline, nullptr);
    vkDestroyPipeline(logicalDevice, graphicsOpaqueDoubleSidedPipeline,
                      nullptr);
    vkDestroyPipeline(logicalDevice, graphicsMaskPipeline, nullptr);
    vkDestroyPipeline(logicalDevice, graphicsMaskDoubleSidedPipeline,
                      nullptr);
    vkDestroyPipeline(logicalDevice, graphicsBlendPipeline, nullptr);
    vkDestroyPipeline(logicalDevice, graphicsBlendDoubleSidedPipeline,
                      nullptr);
    vkDestroyPipeline(logicalDevice, shadowOpaquePipeline, nullptr);
    vkDestroyPipeline(logicalDevice, shadowAlphaPipeline, nullptr);
    vkDestroyPipeline(logicalDevice, grassPipeline, nullptr);
    vkDestroyPipeline(logicalDevice, flowerPipeline, nullptr);
    vkDestroyPipeline(logicalDevice, computePipeline, nullptr);

    vkDestroyPipelineLayout(logicalDevice, graphicsPipelineLayout, nullptr);
    vkDestroyPipelineLayout(logicalDevice, shadowPipelineLayout, nullptr);
    vkDestroyPipelineLayout(logicalDevice, grassPipelineLayout, nullptr);
    vkDestroyPipelineLayout(logicalDevice, flowerPipelineLayout, nullptr);
    vkDestroyPipelineLayout(logicalDevice, computePipelineLayout, nullptr);

    vkDestroyDescriptorSetLayout(logicalDevice, cameraDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(logicalDevice, modelDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(logicalDevice, shadowDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(logicalDevice, environmentDescriptorSetLayout,
                                 nullptr);
    vkDestroyDescriptorSetLayout(logicalDevice, timeDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(logicalDevice, computeDescriptorSetLayout, nullptr);

    vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);

    DestroyEnvironmentResources();
    DestroyShadowResources();
    vkDestroyRenderPass(logicalDevice, renderPass, nullptr);
    vkDestroyRenderPass(logicalDevice, shadowRenderPass, nullptr);
    DestroyFrameResources();
    vkDestroyCommandPool(logicalDevice, computeCommandPool, nullptr);
    vkDestroyCommandPool(logicalDevice, graphicsCommandPool, nullptr);


    vkDestroyPipeline(logicalDevice, skyboxPipeline, nullptr);
    vkDestroyPipelineLayout(logicalDevice, skyboxPipelineLayout, nullptr);
    vkDestroyBuffer(logicalDevice, skyboxVertexBuffer, nullptr);
    vkFreeMemory(logicalDevice, skyboxVertexBufferMemory, nullptr);
}


void Renderer::CreateSkyboxResources() {
    // Skybox cube vertices (36 vertices for 6 faces)
    float skyboxVertices[] = {
        // positions          
        -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

        -1.0f,  1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f,  1.0f
    };

    BufferUtils::CreateBufferFromData(
        device, graphicsCommandPool,
        skyboxVertices, sizeof(skyboxVertices),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        skyboxVertexBuffer, skyboxVertexBufferMemory
    );
}

void Renderer::CreateSkyboxPipeline() {
    VkShaderModule vertShaderModule = ShaderModule::Create("shaders/skybox.vert.spv", logicalDevice);
    VkShaderModule fragShaderModule = ShaderModule::Create("shaders/skybox.frag.spv", logicalDevice);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    VkVertexInputBindingDescription bindingDescription = {};
    bindingDescription.binding = 0;
    bindingDescription.stride = 3 * sizeof(float);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attributeDescription = {};
    attributeDescription.binding = 0;
    attributeDescription.location = 0;
    attributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescription.offset = 0;

    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = 1;
    vertexInputInfo.pVertexAttributeDescriptions = &attributeDescription;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)swapChain->GetVkExtent().width;
    viewport.height = (float)swapChain->GetVkExtent().height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = { 0, 0 };
    scissor.extent = swapChain->GetVkExtent();

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_FALSE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask = 0xF;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkDescriptorSetLayout setLayouts[] = {
        cameraDescriptorSetLayout, environmentDescriptorSetLayout
    };
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 2;
    pipelineLayoutInfo.pSetLayouts = setLayouts;

    if (vkCreatePipelineLayout(logicalDevice, &pipelineLayoutInfo, nullptr, &skyboxPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create skybox pipeline layout");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = skyboxPipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &skyboxPipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create skybox pipeline");
    }

    vkDestroyShaderModule(logicalDevice, vertShaderModule, nullptr);
    vkDestroyShaderModule(logicalDevice, fragShaderModule, nullptr);
}
