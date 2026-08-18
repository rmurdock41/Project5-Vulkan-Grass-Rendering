#pragma once

#include "Device.h"
#include "SwapChain.h"
#include "Scene.h"
#include "Camera.h"
#include "EnvironmentMap.h"

#include <string>
#include <vector>

struct ShadowBufferObject {
    glm::mat4 lightViewProjection;
    // xyz points from the shaded surface toward the directional light.
    glm::vec4 lightDirection;
};

struct EnvironmentSettings {
    std::string path;
    float backgroundIntensity = 1.0f;
    float lightingIntensity = 1.0f;
    float directLightingIntensity = 1.0f;
    float rotationRadians = 0.0f;
    bool visible = true;
};

struct EnvironmentBufferObject {
    // x: background intensity, y: lighting intensity,
    // z: rotation around world Y, w: one when an HDR is loaded.
    glm::vec4 parameters;
    // x: background visible, y: maximum cubemap mip,
    // z: directional-light intensity.
    glm::vec4 options;
};

class Renderer {
public:
    Renderer() = delete;
    Renderer(Device* device, SwapChain* swapChain, Scene* scene,
             Camera* camera, const EnvironmentSettings& environment,
             bool frustumCullingEnabled = true,
             bool shadowCullingEnabled = true);
    ~Renderer();

    void CreateCommandPools();

    void CreateRenderPass();
    void CreateShadowRenderPass();

    void CreateCameraDescriptorSetLayout();
    void CreateModelDescriptorSetLayout();
    void CreateShadowDescriptorSetLayout();
    void CreateEnvironmentDescriptorSetLayout();
    void CreateTimeDescriptorSetLayout();
    void CreateComputeDescriptorSetLayout();

    void CreateDescriptorPool();

    void CreateCameraDescriptorSet();
    void CreateModelDescriptorSets();
    void CreateShadowDescriptorSet();
    void CreateEnvironmentDescriptorSet();
    void CreateGrassDescriptorSets();
    void CreateTimeDescriptorSet();
    void CreateComputeDescriptorSets();

    void CreateGraphicsPipeline();
    void CreateShadowPipeline();
    void CreateGrassPipeline();
    void CreateFlowerPipeline();
    void CreateComputePipeline();

    void CreateFrameResources();
    void CreateShadowResources();
    void DestroyShadowResources();
    void CreateEnvironmentResources();
    void DestroyEnvironmentResources();
    void DestroyFrameResources();
    void RecreateFrameResources();

    void RecordCommandBuffers();
    void RecordCommandBuffer(uint32_t imageIndex);
    void DestroyCommandBuffers();
    void RecordComputeCommandBuffers();

    void Frame();

    uint32_t GetVisibleModelCount() const;
    uint32_t GetCulledModelCount() const;
    uint32_t GetTotalModelCount() const;
    bool IsFrustumCullingEnabled() const;
    uint32_t GetShadowCasterCount() const;
    uint32_t GetShadowCulledModelCount() const;
    bool IsShadowCullingEnabled() const;
    bool IsGpuProfilerSupported() const;
    bool HasGpuProfilerResults() const;
    double GetGraphicsGpuTimeMs() const;
    double GetShadowGpuTimeMs() const;
    double GetMainGpuTimeMs() const;
    void SetEnvironmentRotationDegrees(float rotationDegrees);

    void CreateSkyboxResources();
    void CreateSkyboxPipeline();

private:
    Device* device;
    VkDevice logicalDevice;
    SwapChain* swapChain;
    Scene* scene;
    Camera* camera;
    EnvironmentSettings environmentSettings;
    bool frustumCullingEnabled;
    bool shadowCullingEnabled;
    bool hasReportedCullingStats = false;
    uint32_t visibleModelCount = 0;
    uint32_t culledModelCount = 0;
    std::vector<uint32_t> visibleModelIndices;
    std::vector<uint32_t> visibleOpaqueSingleSidedIndices;
    std::vector<uint32_t> visibleOpaqueDoubleSidedIndices;
    std::vector<uint32_t> visibleMaskSingleSidedIndices;
    std::vector<uint32_t> visibleMaskDoubleSidedIndices;
    std::vector<uint32_t> visibleBlendIndices;
    uint32_t shadowCasterCount = 0;
    uint32_t shadowCulledModelCount = 0;
    uint32_t shadowSubpixelCulledCount = 0;
    uint32_t shadowReceiverCulledCount = 0;
    std::vector<uint32_t> shadowCasterIndices;
    std::vector<uint32_t> shadowOpaqueCasterIndices;
    std::vector<uint32_t> shadowAlphaCasterIndices;

    VkCommandPool graphicsCommandPool;
    VkCommandPool computeCommandPool;

    VkRenderPass renderPass;
    VkRenderPass shadowRenderPass;

    VkDescriptorSetLayout cameraDescriptorSetLayout;
    VkDescriptorSetLayout modelDescriptorSetLayout;
    VkDescriptorSetLayout shadowDescriptorSetLayout;
    VkDescriptorSetLayout environmentDescriptorSetLayout;
    VkDescriptorSetLayout timeDescriptorSetLayout;

    VkDescriptorSetLayout computeDescriptorSetLayout;
    
    VkDescriptorPool descriptorPool;

    std::vector<VkDescriptorSet> cameraDescriptorSets;
    VkDescriptorSet shadowDescriptorSet;
    VkDescriptorSet environmentDescriptorSet;
    std::vector<VkDescriptorSet> modelDescriptorSets;
	std::vector<VkDescriptorSet> grassDescriptorSets;

    std::vector<VkDescriptorSet> computeDescriptorSets;

    VkDescriptorSet timeDescriptorSet;

    VkPipelineLayout graphicsPipelineLayout;
    VkPipelineLayout shadowPipelineLayout;
    VkPipelineLayout grassPipelineLayout;
    VkPipelineLayout flowerPipelineLayout;
    VkPipelineLayout computePipelineLayout;

    VkPipeline graphicsOpaquePipeline;
    VkPipeline graphicsOpaqueDoubleSidedPipeline;
    VkPipeline graphicsMaskPipeline;
    VkPipeline graphicsMaskDoubleSidedPipeline;
    VkPipeline graphicsBlendPipeline;
    VkPipeline graphicsBlendDoubleSidedPipeline;
    VkPipeline shadowOpaquePipeline;
    VkPipeline shadowAlphaPipeline;
    VkPipeline grassPipeline;
    VkPipeline flowerPipeline;
    VkPipeline computePipeline;

    std::vector<VkImageView> imageViews;
    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;
    std::vector<VkFramebuffer> framebuffers;

    static const uint32_t ShadowMapSize = 2048;
    VkFormat shadowDepthFormat;
    VkImage shadowDepthImage;
    VkDeviceMemory shadowDepthImageMemory;
    VkImageView shadowDepthImageView;
    VkSampler shadowSampler;
    VkFramebuffer shadowFramebuffer;
    VkBuffer shadowBuffer;
    VkDeviceMemory shadowBufferMemory;
    ShadowBufferObject shadowBufferObject;

    EnvironmentMap::Resources environmentMap;
    VkBuffer environmentBuffer;
    VkDeviceMemory environmentBufferMemory;
    EnvironmentBufferObject environmentBufferObject;

    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkFence> commandBufferFences;
    std::vector<VkCommandBuffer> computeCommandBuffers;
    std::vector<VkSemaphore> computeFinishedSemaphores;

    enum TimestampQuery : uint32_t {
        TimestampGraphicsBegin = 0,
        TimestampShadowBegin,
        TimestampShadowEnd,
        TimestampMainBegin,
        TimestampMainEnd,
        TimestampGraphicsEnd,
        TimestampQueryCount
    };
    bool gpuProfilerSupported = false;
    bool hasGpuProfilerResults = false;
    uint32_t timestampValidBits = 0;
    double timestampPeriodNanoseconds = 0.0;
    double graphicsGpuTimeMs = 0.0;
    double shadowGpuTimeMs = 0.0;
    double mainGpuTimeMs = 0.0;
    std::vector<VkQueryPool> timestampQueryPools;
    std::vector<bool> timestampQueriesSubmitted;

    void UpdateVisibleModels();
    void CreateTimestampQueryResources();
    void DestroyTimestampQueryResources();
    void ReadTimestampResults(uint32_t imageIndex);
    double TimestampDeltaMilliseconds(uint64_t begin,
                                      uint64_t end) const;

    
    VkPipelineLayout skyboxPipelineLayout;
    VkPipeline skyboxPipeline;
    VkBuffer skyboxVertexBuffer;
    VkDeviceMemory skyboxVertexBufferMemory;
};
