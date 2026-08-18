#include <vulkan/vulkan.h>
#include "Instance.h"
#include "Window.h"
#include "Renderer.h"
#include "Camera.h"
#include "Scene.h"
#include "Image.h"
#include "GltfLoader.h"
#include "GltfModelFactory.h"
#include "SceneConfig.h"
#include "EnvironmentControlPanel.h"
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>

Device* device;
SwapChain* swapChain;
Renderer* renderer;
Camera* camera;

namespace {
    struct LoadedStaticModel {
        Gltf::SceneData scene;
        glm::mat4 transform = glm::mat4(1.0f);
        std::string path;
    };

    std::string LowercaseExtension(const std::string& path) {
        const std::string::size_type dot = path.find_last_of('.');
        if (dot == std::string::npos) {
            return std::string();
        }
        std::string extension = path.substr(dot);
        std::transform(extension.begin(), extension.end(), extension.begin(),
            [](unsigned char value) {
                return static_cast<char>(std::tolower(value));
            });
        return extension;
    }

    glm::mat4 ComposeTransform(const SceneConfig::ModelEntry& entry) {
        glm::mat4 transform = glm::translate(
            glm::mat4(1.0f), entry.position);
        transform = glm::rotate(
            transform, glm::radians(entry.rotationDegrees.x),
            glm::vec3(1.0f, 0.0f, 0.0f));
        transform = glm::rotate(
            transform, glm::radians(entry.rotationDegrees.y),
            glm::vec3(0.0f, 1.0f, 0.0f));
        transform = glm::rotate(
            transform, glm::radians(entry.rotationDegrees.z),
            glm::vec3(0.0f, 0.0f, 1.0f));
        return glm::scale(transform, entry.scale);
    }

    glm::mat4 ComposeGrassTransform(const SceneConfig::GrassEntry& entry) {
        glm::mat4 transform = glm::translate(
            glm::mat4(1.0f), entry.position);
        transform = glm::rotate(
            transform, glm::radians(entry.rotationDegrees.x),
            glm::vec3(1.0f, 0.0f, 0.0f));
        transform = glm::rotate(
            transform, glm::radians(entry.rotationDegrees.y),
            glm::vec3(0.0f, 1.0f, 0.0f));
        transform = glm::rotate(
            transform, glm::radians(entry.rotationDegrees.z),
            glm::vec3(0.0f, 0.0f, 1.0f));
        return glm::scale(transform, entry.scale);
    }

    std::vector<glm::vec3> LoadGrassPositionsOrThrow(
            const SceneConfig::GrassEntry& entry) {
        if (entry.positionsPath.empty()) {
            return {};
        }

        std::ifstream stream(entry.positionsPath,
                             std::ios::binary | std::ios::ate);
        if (!stream) {
            throw std::runtime_error(
                "Cannot open grass positions file: " +
                entry.positionsPath);
        }
        const std::streamoff fileSize = stream.tellg();
        stream.seekg(0, std::ios::beg);

        char magic[4] = {};
        std::uint32_t count = 0;
        stream.read(magic, sizeof(magic));
        stream.read(reinterpret_cast<char*>(&count), sizeof(count));
        if (!stream || std::memcmp(magic, "GRS1", 4) != 0) {
            throw std::runtime_error(
                "Invalid grass positions header in: " +
                entry.positionsPath);
        }
        const std::streamoff expectedSize =
            8 + static_cast<std::streamoff>(count) * 3 * sizeof(float);
        if (count == 0 || fileSize != expectedSize) {
            throw std::runtime_error(
                "Invalid grass positions payload in: " +
                entry.positionsPath);
        }

        const glm::mat4 transform = ComposeGrassTransform(entry);
        std::vector<glm::vec3> positions;
        positions.reserve(count);
        for (std::uint32_t index = 0; index < count; ++index) {
            float xyz[3] = {};
            stream.read(reinterpret_cast<char*>(xyz), sizeof(xyz));
            if (!stream) {
                throw std::runtime_error(
                    "Unexpected end of grass positions file: " +
                    entry.positionsPath);
            }
            positions.push_back(glm::vec3(
                transform * glm::vec4(xyz[0], xyz[1], xyz[2], 1.0f)));
        }
        return positions;
    }

    void LoadGltfOrThrow(const std::string& path,
                         Gltf::SceneData& output) {
        std::string errors;
        std::string warnings;
        if (!Gltf::Loader::Load(path, output, errors, warnings)) {
            throw std::runtime_error(
                "Failed to load static glTF model '" + path + "':\n" +
                warnings + errors);
        }
        if (!warnings.empty()) {
            std::cerr << "glTF warnings for " << path << ":\n" << warnings;
        }
    }

    void resizeCallback(GLFWwindow* window, int width, int height) {
        if (width == 0 || height == 0) return;

        vkDeviceWaitIdle(device->GetVkDevice());
        swapChain->Recreate();
        const VkExtent2D extent = swapChain->GetVkExtent();
        camera->SetAspectRatio(
            static_cast<float>(extent.width) /
            static_cast<float>(extent.height));
        renderer->RecreateFrameResources();
    }

    bool leftMouseDown = false;
    bool rightMouseDown = false;
    double previousX = 0.0;
    double previousY = 0.0;

    void mouseDownCallback(GLFWwindow* window, int button, int action, int mods) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            if (action == GLFW_PRESS) {
                leftMouseDown = true;
                glfwGetCursorPos(window, &previousX, &previousY);
            }
            else if (action == GLFW_RELEASE) {
                leftMouseDown = false;
            }
        } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            if (action == GLFW_PRESS) {
                rightMouseDown = true;
                glfwGetCursorPos(window, &previousX, &previousY);
            }
            else if (action == GLFW_RELEASE) {
                rightMouseDown = false;
            }
        }
    }

    void mouseMoveCallback(GLFWwindow* window, double xPosition, double yPosition) {
        if (leftMouseDown) {
            double sensitivity = 0.5;
            float deltaX = static_cast<float>((previousX - xPosition) * sensitivity);
            float deltaY = static_cast<float>((previousY - yPosition) * sensitivity);

            camera->UpdateOrbit(deltaX, deltaY, 0.0f);

            previousX = xPosition;
            previousY = yPosition;
        } else if (rightMouseDown) {
            double deltaZ = static_cast<float>((previousY - yPosition) * 0.05);

            camera->UpdateOrbit(0.0f, 0.0f, deltaZ);

            previousY = yPosition;
        }
    }

    void keyCallback(GLFWwindow* window, int key, int scancode,
                     int action, int mods) {
        if (key != GLFW_KEY_P || action != GLFW_PRESS || camera == nullptr) {
            return;
        }

        const glm::vec3 position = camera->GetPosition();
        const glm::vec3 target = camera->GetTarget();
        std::ofstream output("camera_capture.json", std::ios::trunc);
        if (!output) {
            std::cerr << "Failed to save camera_capture.json\n";
            return;
        }

        output << std::fixed << std::setprecision(6)
               << "{\n"
               << "  \"camera\": {\n"
               << "    \"position\": [" << position.x << ", "
               << position.y << ", " << position.z << "],\n"
               << "    \"target\": [" << target.x << ", "
               << target.y << ", " << target.z << "],\n"
               << "    \"fov\": " << camera->GetFieldOfViewDegrees()
               << ",\n"
               << "    \"near\": " << camera->GetNearPlane() << ",\n"
               << "    \"far\": " << camera->GetFarPlane() << "\n"
               << "  },\n"
               << "  \"environment\": {\n"
               << "    \"rotationDegrees\": "
               << EnvironmentControlPanel::GetCurrentRotationDegrees()
               << "\n"
               << "  }\n"
               << "}\n";
        output.close();
        std::cout << "Saved current camera to camera_capture.json\n";
    }
}

int RunApplication(int argc, char** argv) {
    static constexpr char* applicationName = "Vulkan Grass Rendering";

    std::string inputPath;
    bool modelOnly = false;
    bool frustumCullingEnabled = true;
    bool shadowCullingEnabled = true;
    bool staticBatchingEnabled = true;
    bool mipmapsEnabled = true;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string value(argv[argument]);
        if (value == "--model-only") {
            modelOnly = true;
        } else if (value == "--no-frustum-culling") {
            frustumCullingEnabled = false;
        } else if (value == "--no-shadow-culling") {
            shadowCullingEnabled = false;
        } else if (value == "--no-static-batching") {
            staticBatchingEnabled = false;
        } else if (value == "--no-mipmaps") {
            mipmapsEnabled = false;
        } else if (inputPath.empty()) {
            inputPath = value;
        } else {
            throw std::runtime_error("Unexpected argument: " + value);
        }
    }

    Image::SetMipmapsEnabled(mipmapsEnabled);

    std::vector<LoadedStaticModel> loadedStaticModels;
    bool showGround = true;
    bool showGrass = true;
    SceneConfig::GrassEntry grassConfig;
    SceneConfig::CameraEntry cameraConfig;
    EnvironmentSettings environmentSettings;
    if (!inputPath.empty() && LowercaseExtension(inputPath) == ".json") {
        SceneConfig::Data config;
        std::string configError;
        if (!SceneConfig::Load(inputPath, config, configError)) {
            throw std::runtime_error(
                "Failed to load scene config '" + inputPath + "':\n" +
                configError);
        }
        showGround = config.showGround;
        showGrass = config.showGrass;
        grassConfig = config.grass;
        cameraConfig = config.camera;
        if (config.environment.enabled) {
            environmentSettings.path = config.environment.path;
            environmentSettings.backgroundIntensity =
                config.environment.backgroundIntensity;
            environmentSettings.lightingIntensity =
                config.environment.lightingIntensity;
            environmentSettings.directLightingIntensity =
                config.environment.directLightingIntensity;
            environmentSettings.rotationRadians =
                glm::radians(config.environment.rotationDegrees);
            environmentSettings.visible = config.environment.visible;
        }
        loadedStaticModels.reserve(config.models.size());
        for (std::size_t i = 0; i < config.models.size(); ++i) {
            LoadedStaticModel loaded;
            loaded.path = config.models[i].path;
            loaded.transform = ComposeTransform(config.models[i]);
            LoadGltfOrThrow(loaded.path, loaded.scene);
            loadedStaticModels.push_back(loaded);
        }
        std::cout << "Scene config loaded " << loadedStaticModels.size()
                  << " model entries.\n";
    } else if (!inputPath.empty()) {
        LoadedStaticModel loaded;
        loaded.path = inputPath;
        loaded.transform = glm::rotate(
            glm::mat4(1.0f), glm::radians(180.0f),
            glm::vec3(0.0f, 1.0f, 0.0f));
        LoadGltfOrThrow(loaded.path, loaded.scene);
        loadedStaticModels.push_back(loaded);
    }
    if (modelOnly) {
        showGround = false;
        showGrass = false;
    }

    InitializeWindow(640, 480, applicationName);

    unsigned int glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    Instance* instance = new Instance(applicationName, glfwExtensionCount, glfwExtensions);

    VkSurfaceKHR surface;
    if (glfwCreateWindowSurface(instance->GetVkInstance(), GetGLFWWindow(), nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface");
    }

    instance->PickPhysicalDevice({ VK_KHR_SWAPCHAIN_EXTENSION_NAME }, QueueFlagBit::GraphicsBit | QueueFlagBit::TransferBit | QueueFlagBit::ComputeBit | QueueFlagBit::PresentBit, surface);

    VkPhysicalDeviceFeatures deviceFeatures = {};
    deviceFeatures.tessellationShader = VK_TRUE;
    deviceFeatures.fillModeNonSolid = VK_TRUE;
    deviceFeatures.samplerAnisotropy = VK_TRUE;

    device = instance->CreateDevice(QueueFlagBit::GraphicsBit | QueueFlagBit::TransferBit | QueueFlagBit::ComputeBit | QueueFlagBit::PresentBit, deviceFeatures);

    swapChain = device->CreateSwapChain(surface, 5);

    const VkExtent2D initialExtent = swapChain->GetVkExtent();
    camera = new Camera(
        device, static_cast<float>(initialExtent.width) /
                    static_cast<float>(initialExtent.height),
        swapChain->GetCount());
    if (cameraConfig.enabled) {
        camera->SetView(cameraConfig.position, cameraConfig.target,
                        cameraConfig.fieldOfViewDegrees,
                        cameraConfig.nearPlane, cameraConfig.farPlane);
    }
    if (modelOnly) {
        camera->UpdateOrbit(0.0f, 0.0f, 7.0f);
    }

    VkCommandPoolCreateInfo transferPoolInfo = {};
    transferPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    // Upload helpers submit their copy/blit work to the graphics queue, so the
    // command pool must belong to that same queue family.
    transferPoolInfo.queueFamilyIndex =
        device->GetInstance()->GetQueueFamilyIndices()[QueueFlags::Graphics];
    transferPoolInfo.flags = 0;

    VkCommandPool transferCommandPool;
    if (vkCreateCommandPool(device->GetVkDevice(), &transferPoolInfo, nullptr, &transferCommandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }

    VkImage grassImage;
    VkDeviceMemory grassImageMemory;
    uint32_t grassMipLevels = 1;
    Image::FromFile(device,
        transferCommandPool,
        "images/grass.jpg",
        VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        grassImage,
        grassImageMemory,
        &grassMipLevels
    );

    VkImage whiteFallbackImage = VK_NULL_HANDLE;
    VkDeviceMemory whiteFallbackMemory = VK_NULL_HANDLE;
    VkImage normalFallbackImage = VK_NULL_HANDLE;
    VkDeviceMemory normalFallbackMemory = VK_NULL_HANDLE;
    VkImage metallicRoughnessFallbackImage = VK_NULL_HANDLE;
    VkDeviceMemory metallicRoughnessFallbackMemory = VK_NULL_HANDLE;
    const unsigned char whitePixel[4] = {255, 255, 255, 255};
    const unsigned char flatNormalPixel[4] = {128, 128, 255, 255};
    // glTF stores roughness in G and metallic in B.
    const unsigned char dielectricRoughPixel[4] = {255, 255, 0, 255};
    Image::FromPixels(device, transferCommandPool, whitePixel, 1, 1,
                      whiteFallbackImage, whiteFallbackMemory);
    Image::FromPixels(device, transferCommandPool, flatNormalPixel, 1, 1,
                      normalFallbackImage, normalFallbackMemory);
    Image::FromPixels(device, transferCommandPool, dielectricRoughPixel, 1, 1,
                      metallicRoughnessFallbackImage,
                      metallicRoughnessFallbackMemory);

    float planeDim = 15.f;
    float halfWidth = planeDim * 0.5f;
    Model* plane = new Model(device, transferCommandPool,
        {
            { { -halfWidth, 0.0f, halfWidth }, { 1.0f, 1.0f, 1.0f },{ 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f, 1.0f } },
            { { halfWidth, 0.0f, halfWidth }, { 1.0f, 1.0f, 1.0f },{ 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f, 1.0f } },
            { { halfWidth, 0.0f, -halfWidth }, { 1.0f, 1.0f, 1.0f },{ 0.0f, 1.0f }, { 0.0f, 1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f, 1.0f } },
            { { -halfWidth, 0.0f, -halfWidth }, { 1.0f, 1.0f, 1.0f },{ 1.0f, 1.0f }, { 0.0f, 1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f, 1.0f } }
        },
        { 0, 1, 2, 2, 3, 0 }
    );
    plane->SetTexture(MaterialTextureSlot::BaseColor, grassImage,
                      VK_FORMAT_R8G8B8A8_SRGB, grassMipLevels);
    plane->SetTexture(MaterialTextureSlot::Normal, normalFallbackImage);
    plane->SetTexture(MaterialTextureSlot::MetallicRoughness,
                      metallicRoughnessFallbackImage);
    plane->SetTexture(MaterialTextureSlot::Specular, whiteFallbackImage);
    plane->SetTexture(MaterialTextureSlot::SpecularColor,
                      whiteFallbackImage);

    std::vector<Gltf::StaticModelResources> staticResources;
    if (!loadedStaticModels.empty()) {
        Gltf::TextureFallbacks fallbacks;
        fallbacks.baseColor = whiteFallbackImage;
        fallbacks.normal = normalFallbackImage;
        fallbacks.metallicRoughness = metallicRoughnessFallbackImage;
        fallbacks.specular = whiteFallbackImage;
        fallbacks.specularColor = whiteFallbackImage;
        staticResources.reserve(loadedStaticModels.size());
        Gltf::StaticBatchingSettings batchingSettings;
        batchingSettings.enabled = staticBatchingEnabled;
        for (std::size_t i = 0; i < loadedStaticModels.size(); ++i) {
            staticResources.push_back(Gltf::CreateStaticModels(
                device, transferCommandPool, loadedStaticModels[i].scene,
                fallbacks, loadedStaticModels[i].transform,
                batchingSettings));
            const Gltf::StaticModelResources& uploaded = staticResources.back();
            uint32_t maximumMipLevels = 1;
            std::size_t mipmappedTextureCount = 0;
            for (std::size_t textureIndex = 0;
                 textureIndex < uploaded.textureMipLevels.size();
                 ++textureIndex) {
                maximumMipLevels = std::max(
                    maximumMipLevels,
                    uploaded.textureMipLevels[textureIndex]);
                if (uploaded.textureMipLevels[textureIndex] > 1) {
                    ++mipmappedTextureCount;
                }
            }
            std::cout << "Uploaded entry " << i << " ("
                      << loadedStaticModels[i].path << "): "
                      << uploaded.sourcePrimitiveCount << " primitives -> "
                      << uploaded.outputBatchCount << " draw batches"
                      << " (saved "
                      << (uploaded.sourcePrimitiveCount -
                          uploaded.outputBatchCount)
                      << ", merged primitives "
                      << uploaded.mergedPrimitiveCount
                      << ", standalone large "
                      << uploaded.standaloneLargePrimitiveCount
                      << ", standalone blend "
                      << uploaded.standaloneBlendPrimitiveCount
                      << ") and "
                      << uploaded.textureImages.size()
                      << " image slots; mipmapped textures "
                      << mipmappedTextureCount << "/"
                      << uploaded.textureImages.size()
                      << ", maximum mip levels "
                      << maximumMipLevels << ".\n";
        }
    }
    
    Blades* blades = nullptr;
    if (showGrass) {
        BladeGenerationSettings bladeSettings;
        bladeSettings.count = grassConfig.count;
        bladeSettings.center = grassConfig.center;
        bladeSettings.extent = grassConfig.extent;
        bladeSettings.minHeight = grassConfig.minHeight;
        bladeSettings.maxHeight = grassConfig.maxHeight;
        bladeSettings.minWidth = grassConfig.minWidth;
        bladeSettings.maxWidth = grassConfig.maxWidth;
        bladeSettings.minStiffness = grassConfig.minStiffness;
        bladeSettings.maxStiffness = grassConfig.maxStiffness;
        bladeSettings.bottomColor = grassConfig.bottomColor;
        bladeSettings.topColor = grassConfig.topColor;
        bladeSettings.rimColor = grassConfig.rimColor;
        bladeSettings.flowersEnabled = grassConfig.flowersEnabled;
        bladeSettings.flowerDensity = grassConfig.flowerDensity;
        bladeSettings.flowerHeightScale = grassConfig.flowerHeightScale;
        bladeSettings.seed = grassConfig.seed;
        const std::vector<glm::vec3> terrainPositions =
            LoadGrassPositionsOrThrow(grassConfig);
        blades = new Blades(device, transferCommandPool, bladeSettings,
                            terrainPositions);
        std::cout << "Created " << blades->GetBladeCount()
                  << " grass blades"
                  << (terrainPositions.empty() ? " on a flat field.\n"
                                               : " on sampled terrain.\n");
        if (blades->GetFlowersEnabled()) {
            std::cout << "Flowers enabled: density="
                      << blades->GetFlowerDensity()
                      << ", heightScale="
                      << blades->GetFlowerHeightScale() << ".\n";
        }
    }

    vkDestroyCommandPool(device->GetVkDevice(), transferCommandPool, nullptr);

    Scene* scene = new Scene(device);
    if (showGround) {
        scene->AddModel(plane);
    }
    for (std::size_t resourceIndex = 0;
         resourceIndex < staticResources.size(); ++resourceIndex) {
        for (std::size_t modelIndex = 0;
             modelIndex < staticResources[resourceIndex].models.size();
             ++modelIndex) {
            scene->AddModel(
                staticResources[resourceIndex].models[modelIndex]);
        }
    }
    if (blades != nullptr) {
        scene->AddBlades(blades);
    }

    renderer = new Renderer(
        device, swapChain, scene, camera, environmentSettings,
        frustumCullingEnabled, shadowCullingEnabled);

    EnvironmentControlPanel::Create(
        glm::degrees(environmentSettings.rotationRadians));

    glfwSetFramebufferSizeCallback(GetGLFWWindow(), resizeCallback);
    glfwSetMouseButtonCallback(GetGLFWWindow(), mouseDownCallback);
    glfwSetCursorPosCallback(GetGLFWWindow(), mouseMoveCallback);
    glfwSetKeyCallback(GetGLFWWindow(), keyCallback);

    auto lastTime = std::chrono::high_resolution_clock::now();
    int frameCount = 0;

    while (!ShouldQuit()) {
        glfwPollEvents();
        float environmentRotationDegrees = 0.0f;
        if (EnvironmentControlPanel::ConsumeRotation(
                environmentRotationDegrees)) {
            renderer->SetEnvironmentRotationDegrees(
                environmentRotationDegrees);
        }
        scene->UpdateTime();
        renderer->Frame();

        frameCount++;
        auto currentTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();

        if (deltaTime >= 0.5f) {
            float fps = frameCount / deltaTime;
            float frameTimeMs = deltaTime / frameCount * 1000.0f;

            std::ostringstream title;
            title << "Vulkan Grass Rendering | FPS: "
                << std::fixed << std::setprecision(1) << fps
                << " | Frame Time: " << std::setprecision(2)
                << frameTimeMs << "ms"
                << " | GLB: " << renderer->GetVisibleModelCount()
                << "/" << renderer->GetTotalModelCount();
            if (renderer->IsFrustumCullingEnabled()) {
                title << " | Culled: "
                      << renderer->GetCulledModelCount();
            } else {
                title << " | Culling: OFF";
            }
            title << " | Shadow: "
                  << renderer->GetShadowCasterCount()
                  << "/" << renderer->GetTotalModelCount();
            if (!renderer->IsShadowCullingEnabled()) {
                title << " (OFF)";
            }
            if (renderer->HasGpuProfilerResults()) {
                title << " | GPU: " << std::setprecision(2)
                      << renderer->GetGraphicsGpuTimeMs() << "ms"
                      << " [Shadow "
                      << renderer->GetShadowGpuTimeMs() << " | Main "
                      << renderer->GetMainGpuTimeMs() << "]";
            } else if (!renderer->IsGpuProfilerSupported()) {
                title << " | GPU profiler: unavailable";
            }

            glfwSetWindowTitle(GetGLFWWindow(), title.str().c_str());

            frameCount = 0;
            lastTime = currentTime;
        }
    }

    vkDeviceWaitIdle(device->GetVkDevice());

    EnvironmentControlPanel::Destroy();

    delete renderer;
    delete scene;
    delete plane;
    for (std::size_t i = 0; i < staticResources.size(); ++i) {
        Gltf::DestroyStaticModels(device, staticResources[i]);
    }
    delete blades;
    vkDestroyImage(device->GetVkDevice(), grassImage, nullptr);
    vkFreeMemory(device->GetVkDevice(), grassImageMemory, nullptr);
    vkDestroyImage(device->GetVkDevice(), whiteFallbackImage, nullptr);
    vkFreeMemory(device->GetVkDevice(), whiteFallbackMemory, nullptr);
    vkDestroyImage(device->GetVkDevice(), normalFallbackImage, nullptr);
    vkFreeMemory(device->GetVkDevice(), normalFallbackMemory, nullptr);
    vkDestroyImage(device->GetVkDevice(), metallicRoughnessFallbackImage,
                   nullptr);
    vkFreeMemory(device->GetVkDevice(), metallicRoughnessFallbackMemory,
                 nullptr);
    delete camera;
    delete swapChain;
    delete device;
    vkDestroySurfaceKHR(instance->GetVkInstance(), surface, nullptr);
    delete instance;
    DestroyWindow();
    return 0;
}

int main(int argc, char** argv) {
    try {
        return RunApplication(argc, argv);
    } catch (const std::exception& exception) {
        std::cerr << "Fatal error: " << exception.what() << '\n';
    } catch (...) {
        std::cerr << "Fatal error: unknown exception\n";
    }
    return EXIT_FAILURE;
}
