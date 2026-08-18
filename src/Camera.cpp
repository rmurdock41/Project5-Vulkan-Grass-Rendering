#include <iostream>
#include <cmath>
#include <stdexcept>

#define GLM_FORCE_RADIANS
// Use Vulkan depth range of 0.0 to 1.0 instead of OpenGL
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/matrix_transform.hpp>

#include "Camera.h"
#include "BufferUtils.h"

Camera::Camera(Device* device, float aspectRatio,
               uint32_t bufferedFrameCount)
    : device(device), aspectRatio(aspectRatio),
      fieldOfViewDegrees(45.0f), nearPlane(0.1f), farPlane(100.0f),
      target(0.0f, 1.0f, 0.0f) {
    r = 10.0f;
    theta = 0.0f;
    phi = 0.0f;
    UpdateMatrices();

    bufferedFrameCount = glm::max(bufferedFrameCount, 1u);
    buffers.resize(bufferedFrameCount, VK_NULL_HANDLE);
    bufferMemories.resize(bufferedFrameCount, VK_NULL_HANDLE);
    mappedData.resize(bufferedFrameCount, nullptr);
    for (uint32_t frameIndex = 0; frameIndex < bufferedFrameCount;
         ++frameIndex) {
        BufferUtils::CreateBuffer(
            device, sizeof(CameraBufferObject),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            buffers[frameIndex], bufferMemories[frameIndex]);
        if (vkMapMemory(device->GetVkDevice(), bufferMemories[frameIndex], 0,
                        sizeof(CameraBufferObject), 0,
                        &mappedData[frameIndex]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to map camera buffer");
        }
        Upload(frameIndex);
    }
}

void Camera::UpdateMatrices() {
    const float thetaRadians = glm::radians(theta);
    const float phiRadians = glm::radians(phi);
    const float horizontalRadius = r * std::cos(phiRadians);
    const glm::vec3 position = target + glm::vec3(
        horizontalRadius * std::sin(thetaRadians),
        r * std::sin(phiRadians),
        horizontalRadius * std::cos(thetaRadians));
    cameraBufferObject.viewMatrix = glm::lookAt(
        position, target, glm::vec3(0.0f, 1.0f, 0.0f));
    cameraBufferObject.projectionMatrix = glm::perspective(
        glm::radians(fieldOfViewDegrees), aspectRatio,
        nearPlane, farPlane);
    cameraBufferObject.projectionMatrix[1][1] *= -1.0f;
}

VkBuffer Camera::GetBuffer(uint32_t frameIndex) const {
    return buffers.at(frameIndex);
}

uint32_t Camera::GetBufferCount() const {
    return static_cast<uint32_t>(buffers.size());
}

void Camera::Upload(uint32_t frameIndex) {
    memcpy(mappedData.at(frameIndex), &cameraBufferObject,
           sizeof(CameraBufferObject));
}

const glm::mat4& Camera::GetViewMatrix() const {
    return cameraBufferObject.viewMatrix;
}

const glm::mat4& Camera::GetProjectionMatrix() const {
    return cameraBufferObject.projectionMatrix;
}

glm::vec3 Camera::GetPosition() const {
    const float thetaRadians = glm::radians(theta);
    const float phiRadians = glm::radians(phi);
    const float horizontalRadius = r * std::cos(phiRadians);
    return target + glm::vec3(
        horizontalRadius * std::sin(thetaRadians),
        r * std::sin(phiRadians),
        horizontalRadius * std::cos(thetaRadians));
}

const glm::vec3& Camera::GetTarget() const {
    return target;
}

float Camera::GetFieldOfViewDegrees() const {
    return fieldOfViewDegrees;
}

float Camera::GetNearPlane() const {
    return nearPlane;
}

float Camera::GetFarPlane() const {
    return farPlane;
}

void Camera::SetView(const glm::vec3& position,
                     const glm::vec3& newTarget,
                     float newFieldOfViewDegrees,
                     float newNearPlane, float newFarPlane) {
    target = newTarget;
    fieldOfViewDegrees = newFieldOfViewDegrees;
    nearPlane = newNearPlane;
    farPlane = newFarPlane;

    const glm::vec3 offset = position - target;
    r = glm::max(glm::length(offset), 0.001f);
    theta = glm::degrees(std::atan2(offset.x, offset.z));
    phi = glm::degrees(std::asin(glm::clamp(offset.y / r,
                                            -1.0f, 1.0f)));
    UpdateMatrices();
}

void Camera::SetAspectRatio(float aspectRatio) {
    if (aspectRatio <= 0.0f) {
        return;
    }
    this->aspectRatio = aspectRatio;
    UpdateMatrices();
}

void Camera::UpdateOrbit(float deltaX, float deltaY, float deltaZ) {
    theta += deltaX;
    phi = glm::clamp(phi + deltaY, -89.0f, 89.0f);
    r = glm::clamp(r - deltaZ, 0.1f, 500.0f);
    UpdateMatrices();
}

Camera::~Camera() {
    for (std::size_t frameIndex = 0; frameIndex < buffers.size();
         ++frameIndex) {
        vkUnmapMemory(device->GetVkDevice(), bufferMemories[frameIndex]);
        vkDestroyBuffer(device->GetVkDevice(), buffers[frameIndex], nullptr);
        vkFreeMemory(device->GetVkDevice(), bufferMemories[frameIndex],
                     nullptr);
    }
}
