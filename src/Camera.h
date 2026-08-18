
#pragma once

#include <glm/glm.hpp>
#include <vector>
#include "Device.h"

struct CameraBufferObject {
  glm::mat4 viewMatrix;
  glm::mat4 projectionMatrix;
};

class Camera {
private:
    Device* device;
    
    CameraBufferObject cameraBufferObject;
    
    std::vector<VkBuffer> buffers;
    std::vector<VkDeviceMemory> bufferMemories;
    std::vector<void*> mappedData;

    float r, theta, phi;
    float aspectRatio;
    float fieldOfViewDegrees;
    float nearPlane;
    float farPlane;
    glm::vec3 target;

    void UpdateMatrices();

public:
    Camera(Device* device, float aspectRatio, uint32_t bufferedFrameCount);
    ~Camera();

    VkBuffer GetBuffer(uint32_t frameIndex) const;
    uint32_t GetBufferCount() const;
    void Upload(uint32_t frameIndex);
    const glm::mat4& GetViewMatrix() const;
    const glm::mat4& GetProjectionMatrix() const;
    glm::vec3 GetPosition() const;
    const glm::vec3& GetTarget() const;
    float GetFieldOfViewDegrees() const;
    float GetNearPlane() const;
    float GetFarPlane() const;

    void SetView(const glm::vec3& position, const glm::vec3& target,
                 float fieldOfViewDegrees, float nearPlane,
                 float farPlane);
    void SetAspectRatio(float aspectRatio);
    void UpdateOrbit(float deltaX, float deltaY, float deltaZ);
};
