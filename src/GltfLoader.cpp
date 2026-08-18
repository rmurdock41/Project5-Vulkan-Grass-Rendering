#include "GltfLoader.h"

#include <tinygltf/tiny_gltf.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

bool PreserveEncodedImage(tinygltf::Image* image, int, std::string*,
                          std::string*, int, int,
                          const unsigned char* bytes, int size, void*) {
    if (image == nullptr || bytes == nullptr || size <= 0) {
        return false;
    }
    image->image.assign(bytes, bytes + size);
    return true;
}

glm::mat4 GetLocalTransform(const tinygltf::Node& node) {
    if (node.matrix.size() == 16) {
        return glm::make_mat4(node.matrix.data());
    }

    glm::mat4 transform(1.0f);
    if (node.translation.size() == 3) {
        transform = glm::translate(
            transform,
            glm::vec3(static_cast<float>(node.translation[0]),
                      static_cast<float>(node.translation[1]),
                      static_cast<float>(node.translation[2])));
    }
    if (node.rotation.size() == 4) {
        const glm::quat rotation(
            static_cast<float>(node.rotation[3]),
            static_cast<float>(node.rotation[0]),
            static_cast<float>(node.rotation[1]),
            static_cast<float>(node.rotation[2]));
        transform *= glm::mat4_cast(rotation);
    }
    if (node.scale.size() == 3) {
        transform = glm::scale(
            transform,
            glm::vec3(static_cast<float>(node.scale[0]),
                      static_cast<float>(node.scale[1]),
                      static_cast<float>(node.scale[2])));
    }
    return transform;
}

std::size_t ComponentSize(int componentType) {
    switch (componentType) {
    case TINYGLTF_COMPONENT_TYPE_BYTE:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        return 1;
    case TINYGLTF_COMPONENT_TYPE_SHORT:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        return 2;
    case TINYGLTF_COMPONENT_TYPE_INT:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
    case TINYGLTF_COMPONENT_TYPE_FLOAT:
        return 4;
    case TINYGLTF_COMPONENT_TYPE_DOUBLE:
        return 8;
    default:
        return 0;
    }
}

std::size_t ComponentCount(int type) {
    switch (type) {
    case TINYGLTF_TYPE_SCALAR: return 1;
    case TINYGLTF_TYPE_VEC2: return 2;
    case TINYGLTF_TYPE_VEC3: return 3;
    case TINYGLTF_TYPE_VEC4: return 4;
    case TINYGLTF_TYPE_MAT2: return 4;
    case TINYGLTF_TYPE_MAT3: return 9;
    case TINYGLTF_TYPE_MAT4: return 16;
    default: return 0;
    }
}

double ReadComponent(const unsigned char* data, int componentType,
                     bool normalized) {
    switch (componentType) {
    case TINYGLTF_COMPONENT_TYPE_BYTE: {
        int8_t value;
        std::memcpy(&value, data, sizeof(value));
        return normalized ? std::max(-1.0, value / 127.0) : value;
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
        uint8_t value;
        std::memcpy(&value, data, sizeof(value));
        return normalized ? value / 255.0 : value;
    }
    case TINYGLTF_COMPONENT_TYPE_SHORT: {
        int16_t value;
        std::memcpy(&value, data, sizeof(value));
        return normalized ? std::max(-1.0, value / 32767.0) : value;
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
        uint16_t value;
        std::memcpy(&value, data, sizeof(value));
        return normalized ? value / 65535.0 : value;
    }
    case TINYGLTF_COMPONENT_TYPE_INT: {
        int32_t value;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
        uint32_t value;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }
    case TINYGLTF_COMPONENT_TYPE_FLOAT: {
        float value;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }
    case TINYGLTF_COMPONENT_TYPE_DOUBLE: {
        double value;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }
    default:
        throw std::runtime_error("Unsupported glTF accessor component type");
    }
}

bool GetAccessorLayout(const tinygltf::Model& model, int accessorIndex,
                       const tinygltf::Accessor*& accessor,
                       const unsigned char*& data, std::size_t& stride,
                       std::string& errors) {
    if (accessorIndex < 0 ||
        accessorIndex >= static_cast<int>(model.accessors.size())) {
        errors += "Invalid accessor index.\n";
        return false;
    }
    accessor = &model.accessors[accessorIndex];
    if (accessor->bufferView < 0 ||
        accessor->bufferView >= static_cast<int>(model.bufferViews.size())) {
        errors += "Sparse-only or missing buffer view is not supported.\n";
        return false;
    }
    const tinygltf::BufferView& view = model.bufferViews[accessor->bufferView];
    if (view.buffer < 0 || view.buffer >= static_cast<int>(model.buffers.size())) {
        errors += "Invalid buffer index in accessor.\n";
        return false;
    }
    const tinygltf::Buffer& buffer = model.buffers[view.buffer];
    const std::size_t elementSize =
        ComponentSize(accessor->componentType) * ComponentCount(accessor->type);
    stride = accessor->ByteStride(view);
    if (stride == 0) {
        stride = elementSize;
    }
    const std::size_t offset = view.byteOffset + accessor->byteOffset;
    const std::size_t required = accessor->count == 0
        ? offset
        : offset + stride * (accessor->count - 1) + elementSize;
    if (elementSize == 0 || required > buffer.data.size()) {
        errors += "Accessor points outside its buffer.\n";
        return false;
    }
    data = buffer.data.data() + offset;
    return true;
}

bool ReadVectorAttribute(const tinygltf::Model& model, int accessorIndex,
                         std::size_t expectedComponents,
                         std::vector<float>& values,
                         std::string& errors, std::string& warnings) {
    const tinygltf::Accessor* accessor = nullptr;
    const unsigned char* data = nullptr;
    std::size_t stride = 0;
    if (!GetAccessorLayout(model, accessorIndex, accessor, data, stride, errors)) {
        return false;
    }
    if (ComponentCount(accessor->type) != expectedComponents) {
        errors += "Accessor has an unexpected vector width.\n";
        return false;
    }
    if (accessor->sparse.isSparse) {
        warnings += "Sparse accessor overrides are not applied.\n";
    }

    const std::size_t componentSize = ComponentSize(accessor->componentType);
    values.resize(accessor->count * expectedComponents);
    for (std::size_t i = 0; i < accessor->count; ++i) {
        const unsigned char* element = data + i * stride;
        for (std::size_t c = 0; c < expectedComponents; ++c) {
            values[i * expectedComponents + c] = static_cast<float>(
                ReadComponent(element + c * componentSize,
                              accessor->componentType, accessor->normalized));
        }
    }
    return true;
}

bool ReadIndices(const tinygltf::Model& model, int accessorIndex,
                 std::vector<uint32_t>& indices, std::string& errors,
                 std::string& warnings) {
    const tinygltf::Accessor* accessor = nullptr;
    const unsigned char* data = nullptr;
    std::size_t stride = 0;
    if (!GetAccessorLayout(model, accessorIndex, accessor, data, stride, errors)) {
        return false;
    }
    if (accessor->type != TINYGLTF_TYPE_SCALAR) {
        errors += "Index accessor is not scalar.\n";
        return false;
    }
    if (accessor->sparse.isSparse) {
        warnings += "Sparse index accessor overrides are not applied.\n";
    }

    indices.resize(accessor->count);
    for (std::size_t i = 0; i < accessor->count; ++i) {
        const double value = ReadComponent(
            data + i * stride, accessor->componentType, false);
        if (value < 0.0 || value > std::numeric_limits<uint32_t>::max()) {
            errors += "Index value is outside uint32 range.\n";
            return false;
        }
        indices[i] = static_cast<uint32_t>(value);
    }
    return true;
}

int ReadTextureIndex(const tinygltf::Value::Object& object,
                     const std::string& key) {
    const tinygltf::Value::Object::const_iterator valueIt = object.find(key);
    if (valueIt == object.end() || !valueIt->second.IsObject()) {
        return -1;
    }
    const tinygltf::Value::Object& textureObject =
        valueIt->second.Get<tinygltf::Value::Object>();
    const tinygltf::Value::Object::const_iterator indexIt =
        textureObject.find("index");
    if (indexIt == textureObject.end() || !indexIt->second.IsInt()) {
        return -1;
    }
    return indexIt->second.Get<int>();
}

void ReadSpecularExtension(const tinygltf::Material& source,
                           Gltf::MaterialData& destination) {
    const std::map<std::string, tinygltf::Value>::const_iterator extensionIt =
        source.extensions.find("KHR_materials_specular");
    if (extensionIt == source.extensions.end() ||
        !extensionIt->second.IsObject()) {
        return;
    }
    const tinygltf::Value::Object& object =
        extensionIt->second.Get<tinygltf::Value::Object>();

    const tinygltf::Value::Object::const_iterator factorIt =
        object.find("specularFactor");
    if (factorIt != object.end() && factorIt->second.IsNumber()) {
        destination.specularFactor =
            static_cast<float>(factorIt->second.GetNumberAsDouble());
    }

    const tinygltf::Value::Object::const_iterator colorIt =
        object.find("specularColorFactor");
    if (colorIt != object.end() && colorIt->second.IsArray()) {
        const tinygltf::Value::Array& array =
            colorIt->second.Get<tinygltf::Value::Array>();
        if (array.size() == 3 && array[0].IsNumber() &&
            array[1].IsNumber() && array[2].IsNumber()) {
            destination.specularColorFactor = glm::vec3(
                static_cast<float>(array[0].GetNumberAsDouble()),
                static_cast<float>(array[1].GetNumberAsDouble()),
                static_cast<float>(array[2].GetNumberAsDouble()));
        }
    }
    destination.specularTexture = ReadTextureIndex(object, "specularTexture");
    destination.specularColorTexture =
        ReadTextureIndex(object, "specularColorTexture");
}

void UpdateBounds(const glm::mat4& transform, const glm::vec3& position,
                  Gltf::SceneData& scene) {
    const glm::vec3 worldPosition = glm::vec3(transform * glm::vec4(position, 1.0f));
    if (!scene.hasBounds) {
        scene.boundsMin = worldPosition;
        scene.boundsMax = worldPosition;
        scene.hasBounds = true;
        return;
    }
    scene.boundsMin = glm::min(scene.boundsMin, worldPosition);
    scene.boundsMax = glm::max(scene.boundsMax, worldPosition);
}

bool AppendPrimitive(const tinygltf::Model& model, const tinygltf::Node& node,
                     const tinygltf::Mesh& mesh,
                     const tinygltf::Primitive& source,
                     const glm::mat4& transform, Gltf::SceneData& scene,
                     std::string& errors, std::string& warnings) {
    const int mode = source.mode < 0 ? TINYGLTF_MODE_TRIANGLES : source.mode;
    if (mode != TINYGLTF_MODE_TRIANGLES) {
        warnings += "Skipped a non-triangle primitive in mesh '" + mesh.name + "'.\n";
        return true;
    }

    const std::map<std::string, int>::const_iterator positionIt =
        source.attributes.find("POSITION");
    if (positionIt == source.attributes.end()) {
        errors += "Mesh '" + mesh.name + "' has a primitive without POSITION.\n";
        return false;
    }

    std::vector<float> positions;
    if (!ReadVectorAttribute(model, positionIt->second, 3, positions,
                             errors, warnings)) {
        return false;
    }
    const std::size_t vertexCount = positions.size() / 3;

    Gltf::PrimitiveData primitive;
    primitive.nodeName = node.name;
    primitive.meshName = mesh.name;
    primitive.materialIndex = source.material;
    primitive.nodeTransform = transform;
    primitive.vertices.resize(vertexCount);

    for (std::size_t i = 0; i < vertexCount; ++i) {
        primitive.vertices[i].position = glm::vec3(
            positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
        UpdateBounds(transform, primitive.vertices[i].position, scene);
    }

    const std::map<std::string, int>::const_iterator normalIt =
        source.attributes.find("NORMAL");
    if (normalIt != source.attributes.end()) {
        std::vector<float> normals;
        if (!ReadVectorAttribute(model, normalIt->second, 3, normals,
                                 errors, warnings) ||
            normals.size() / 3 != vertexCount) {
            errors += "NORMAL count does not match POSITION count.\n";
            return false;
        }
        primitive.hasNormals = true;
        for (std::size_t i = 0; i < vertexCount; ++i) {
            primitive.vertices[i].normal = glm::vec3(
                normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]);
        }
    }

    const std::map<std::string, int>::const_iterator texCoordIt =
        source.attributes.find("TEXCOORD_0");
    if (texCoordIt != source.attributes.end()) {
        std::vector<float> texCoords;
        if (!ReadVectorAttribute(model, texCoordIt->second, 2, texCoords,
                                 errors, warnings) ||
            texCoords.size() / 2 != vertexCount) {
            errors += "TEXCOORD_0 count does not match POSITION count.\n";
            return false;
        }
        primitive.hasTexCoords = true;
        for (std::size_t i = 0; i < vertexCount; ++i) {
            primitive.vertices[i].texCoord = glm::vec2(
                texCoords[i * 2], texCoords[i * 2 + 1]);
        }
    }

    const std::map<std::string, int>::const_iterator tangentIt =
        source.attributes.find("TANGENT");
    if (tangentIt != source.attributes.end()) {
        std::vector<float> tangents;
        if (!ReadVectorAttribute(model, tangentIt->second, 4, tangents,
                                 errors, warnings) ||
            tangents.size() / 4 != vertexCount) {
            errors += "TANGENT count does not match POSITION count.\n";
            return false;
        }
        primitive.hasTangents = true;
        for (std::size_t i = 0; i < vertexCount; ++i) {
            primitive.vertices[i].tangent = glm::vec4(
                tangents[i * 4], tangents[i * 4 + 1], tangents[i * 4 + 2],
                tangents[i * 4 + 3]);
        }
    }

    if (source.indices >= 0) {
        if (!ReadIndices(model, source.indices, primitive.indices,
                         errors, warnings)) {
            return false;
        }
    } else {
        primitive.indices.resize(vertexCount);
        for (std::size_t i = 0; i < vertexCount; ++i) {
            primitive.indices[i] = static_cast<uint32_t>(i);
        }
    }

    if (primitive.indices.size() % 3 != 0) {
        warnings += "Triangle primitive has an index count not divisible by three.\n";
    }
    for (std::size_t i = 0; i < primitive.indices.size(); ++i) {
        if (primitive.indices[i] >= vertexCount) {
            errors += "Primitive contains an index outside its vertex range.\n";
            return false;
        }
    }
    scene.primitives.push_back(std::move(primitive));
    return true;
}

bool TraverseNode(const tinygltf::Model& model, int nodeIndex,
                  const glm::mat4& parentTransform, Gltf::SceneData& scene,
                  std::string& errors, std::string& warnings) {
    if (nodeIndex < 0 || nodeIndex >= static_cast<int>(model.nodes.size())) {
        errors += "Scene references an invalid node index.\n";
        return false;
    }
    const tinygltf::Node& node = model.nodes[nodeIndex];
    const glm::mat4 transform = parentTransform * GetLocalTransform(node);

    if (node.mesh >= 0) {
        if (node.mesh >= static_cast<int>(model.meshes.size())) {
            errors += "Node references an invalid mesh index.\n";
            return false;
        }
        const tinygltf::Mesh& mesh = model.meshes[node.mesh];
        for (std::size_t i = 0; i < mesh.primitives.size(); ++i) {
            if (!AppendPrimitive(model, node, mesh, mesh.primitives[i],
                                 transform, scene, errors, warnings)) {
                return false;
            }
        }
    }

    for (std::size_t i = 0; i < node.children.size(); ++i) {
        if (!TraverseNode(model, node.children[i], transform, scene,
                          errors, warnings)) {
            return false;
        }
    }
    return true;
}

} // namespace

namespace Gltf {

bool Loader::Load(const std::string& path, SceneData& output,
                  std::string& errors, std::string& warnings) {
    output = SceneData();
    output.sourcePath = path;
    errors.clear();
    warnings.clear();

    tinygltf::TinyGLTF loader;
    loader.SetImageLoader(PreserveEncodedImage, nullptr);
    tinygltf::Model model;

    bool loaded = false;
    const std::size_t extensionPosition = path.find_last_of('.');
    const std::string extension = extensionPosition == std::string::npos
        ? std::string()
        : path.substr(extensionPosition);
    if (extension == ".glb" || extension == ".GLB") {
        loaded = loader.LoadBinaryFromFile(&model, &errors, &warnings, path);
    } else {
        loaded = loader.LoadASCIIFromFile(&model, &errors, &warnings, path);
    }
    if (!loaded) {
        return false;
    }

    output.sourceNodeCount = model.nodes.size();
    output.sourceMeshCount = model.meshes.size();

    output.materials.reserve(model.materials.size());
    for (std::size_t i = 0; i < model.materials.size(); ++i) {
        const tinygltf::Material& source = model.materials[i];
        MaterialData material;
        material.name = source.name;
        const std::vector<double>& factor =
            source.pbrMetallicRoughness.baseColorFactor;
        if (factor.size() == 4) {
            material.baseColorFactor = glm::vec4(
                static_cast<float>(factor[0]), static_cast<float>(factor[1]),
                static_cast<float>(factor[2]), static_cast<float>(factor[3]));
        }
        material.metallicFactor =
            static_cast<float>(source.pbrMetallicRoughness.metallicFactor);
        material.roughnessFactor =
            static_cast<float>(source.pbrMetallicRoughness.roughnessFactor);
        material.baseColorTexture =
            source.pbrMetallicRoughness.baseColorTexture.index;
        material.metallicRoughnessTexture =
            source.pbrMetallicRoughness.metallicRoughnessTexture.index;
        material.normalTexture = source.normalTexture.index;
        material.normalScale = static_cast<float>(source.normalTexture.scale);
        material.alphaMode = source.alphaMode.empty() ? "OPAQUE" : source.alphaMode;
        material.alphaCutoff = static_cast<float>(source.alphaCutoff);
        material.doubleSided = source.doubleSided;
        ReadSpecularExtension(source, material);
        output.materials.push_back(material);
    }

    output.textures.reserve(model.textures.size());
    for (std::size_t i = 0; i < model.textures.size(); ++i) {
        TextureData texture;
        texture.name = model.textures[i].name;
        texture.imageIndex = model.textures[i].source;
        texture.samplerIndex = model.textures[i].sampler;
        output.textures.push_back(texture);
    }

    output.images.reserve(model.images.size());
    for (std::size_t i = 0; i < model.images.size(); ++i) {
        ImageData image;
        image.name = model.images[i].name;
        image.uri = model.images[i].uri;
        image.mimeType = model.images[i].mimeType;
        image.encodedBytes = model.images[i].image;
        output.images.push_back(std::move(image));
    }

    const int sceneIndex = model.defaultScene >= 0
        ? model.defaultScene
        : (model.scenes.empty() ? -1 : 0);
    if (sceneIndex < 0 || sceneIndex >= static_cast<int>(model.scenes.size())) {
        errors += "glTF file does not contain a valid scene.\n";
        return false;
    }

    const tinygltf::Scene& rootScene = model.scenes[sceneIndex];
    for (std::size_t i = 0; i < rootScene.nodes.size(); ++i) {
        if (!TraverseNode(model, rootScene.nodes[i], glm::mat4(1.0f),
                          output, errors, warnings)) {
            return false;
        }
    }
    return true;
}

void Loader::PrintSummary(const SceneData& scene, std::ostream& output) {
    std::size_t vertexCount = 0;
    std::size_t indexCount = 0;
    std::size_t missingNormals = 0;
    std::size_t missingTexCoords = 0;
    std::size_t missingTangents = 0;
    std::size_t texturedMaterials = 0;
    std::size_t normalMappedMaterials = 0;
    std::size_t roughnessMappedMaterials = 0;
    std::size_t specularMappedMaterials = 0;
    std::size_t encodedImageBytes = 0;

    for (std::size_t i = 0; i < scene.primitives.size(); ++i) {
        vertexCount += scene.primitives[i].vertices.size();
        indexCount += scene.primitives[i].indices.size();
        missingNormals += scene.primitives[i].hasNormals ? 0 : 1;
        missingTexCoords += scene.primitives[i].hasTexCoords ? 0 : 1;
        missingTangents += scene.primitives[i].hasTangents ? 0 : 1;
    }
    for (std::size_t i = 0; i < scene.materials.size(); ++i) {
        texturedMaterials += scene.materials[i].baseColorTexture >= 0 ? 1 : 0;
        normalMappedMaterials += scene.materials[i].normalTexture >= 0 ? 1 : 0;
        roughnessMappedMaterials +=
            scene.materials[i].metallicRoughnessTexture >= 0 ? 1 : 0;
        specularMappedMaterials +=
            (scene.materials[i].specularTexture >= 0 ||
             scene.materials[i].specularColorTexture >= 0) ? 1 : 0;
    }
    for (std::size_t i = 0; i < scene.images.size(); ++i) {
        encodedImageBytes += scene.images[i].encodedBytes.size();
    }

    output << "GLB validation summary\n"
           << "  source: " << scene.sourcePath << "\n"
           << "  nodes: " << scene.sourceNodeCount << "\n"
           << "  meshes: " << scene.sourceMeshCount << "\n"
           << "  triangle primitives: " << scene.primitives.size() << "\n"
           << "  vertices: " << vertexCount << "\n"
           << "  indices: " << indexCount << "\n"
           << "  triangles: " << indexCount / 3 << "\n"
           << "  materials: " << scene.materials.size() << "\n"
           << "    base-color textures: " << texturedMaterials << "\n"
           << "    normal textures: " << normalMappedMaterials << "\n"
           << "    metallic-roughness textures: " << roughnessMappedMaterials << "\n"
           << "    specular textures: " << specularMappedMaterials << "\n"
           << "  textures: " << scene.textures.size() << "\n"
           << "  images: " << scene.images.size() << " ("
           << encodedImageBytes << " encoded bytes)\n"
           << "  primitives missing normals: " << missingNormals << "\n"
           << "  primitives missing TEXCOORD_0: " << missingTexCoords << "\n";
    output << "  primitives requiring generated tangents: "
           << missingTangents << "\n";

    if (scene.hasBounds) {
        const glm::vec3 extent = scene.boundsMax - scene.boundsMin;
        output << std::fixed << std::setprecision(4)
               << "  bounds min: [" << scene.boundsMin.x << ", "
               << scene.boundsMin.y << ", " << scene.boundsMin.z << "]\n"
               << "  bounds max: [" << scene.boundsMax.x << ", "
               << scene.boundsMax.y << ", " << scene.boundsMax.z << "]\n"
               << "  extent: [" << extent.x << ", " << extent.y << ", "
               << extent.z << "]\n";
    }
}

} // namespace Gltf
