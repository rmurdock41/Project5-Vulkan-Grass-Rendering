#pragma once

#include <glm/glm.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace Gltf {

struct VertexData {
    glm::vec3 position = glm::vec3(0.0f);
    glm::vec3 normal = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec2 texCoord = glm::vec2(0.0f);
    glm::vec4 tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
};

struct PrimitiveData {
    std::string nodeName;
    std::string meshName;
    int materialIndex = -1;
    glm::mat4 nodeTransform = glm::mat4(1.0f);
    std::vector<VertexData> vertices;
    std::vector<uint32_t> indices;
    bool hasNormals = false;
    bool hasTexCoords = false;
    bool hasTangents = false;
};

struct MaterialData {
    std::string name;
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    float specularFactor = 1.0f;
    glm::vec3 specularColorFactor = glm::vec3(1.0f);
    int baseColorTexture = -1;
    int normalTexture = -1;
    float normalScale = 1.0f;
    int metallicRoughnessTexture = -1;
    int specularTexture = -1;
    int specularColorTexture = -1;
    std::string alphaMode = "OPAQUE";
    float alphaCutoff = 0.5f;
    bool doubleSided = false;
};

struct TextureData {
    std::string name;
    int imageIndex = -1;
    int samplerIndex = -1;
};

struct ImageData {
    std::string name;
    std::string uri;
    std::string mimeType;
    std::vector<unsigned char> encodedBytes;
};

struct SceneData {
    std::string sourcePath;
    std::size_t sourceNodeCount = 0;
    std::size_t sourceMeshCount = 0;
    std::vector<PrimitiveData> primitives;
    std::vector<MaterialData> materials;
    std::vector<TextureData> textures;
    std::vector<ImageData> images;
    glm::vec3 boundsMin = glm::vec3(0.0f);
    glm::vec3 boundsMax = glm::vec3(0.0f);
    bool hasBounds = false;
};

class Loader {
public:
    static bool Load(const std::string& path, SceneData& output,
                     std::string& errors, std::string& warnings);
    static void PrintSummary(const SceneData& scene, std::ostream& output);
};

} // namespace Gltf
