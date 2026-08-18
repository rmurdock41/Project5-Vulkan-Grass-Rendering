#include "SceneConfig.h"

#include "tinygltf/json.hpp"

#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace SceneConfig {

namespace {

std::string ParentDirectory(const std::string& path) {
    const std::string::size_type separator = path.find_last_of("/\\");
    if (separator == std::string::npos) {
        return ".";
    }
    if (separator == 0) {
        return path.substr(0, 1);
    }
    return path.substr(0, separator);
}

bool IsAbsolutePath(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    if (path[0] == '/' || path[0] == '\\') {
        return true;
    }
    return path.size() > 1 && path[1] == ':';
}

std::string ResolvePath(const std::string& baseDirectory,
                        const std::string& path) {
    if (IsAbsolutePath(path)) {
        return path;
    }
    return baseDirectory + "/" + path;
}

glm::vec3 ReadVec3(const nlohmann::json& object, const char* key,
                   const glm::vec3& defaultValue,
                   std::size_t modelIndex) {
    const nlohmann::json::const_iterator value = object.find(key);
    if (value == object.end()) {
        return defaultValue;
    }
    if (!value->is_array() || value->size() != 3) {
        throw std::runtime_error(
            "models[" + std::to_string(modelIndex) + "]." + key +
            " must be an array of three numbers");
    }
    for (std::size_t component = 0; component < 3; ++component) {
        if (!(*value)[component].is_number()) {
            throw std::runtime_error(
                "models[" + std::to_string(modelIndex) + "]." + key +
                " must contain only numbers");
        }
    }
    return glm::vec3(
        (*value)[0].get<float>(),
        (*value)[1].get<float>(),
        (*value)[2].get<float>());
}

glm::vec3 ReadObjectVec3(const nlohmann::json& object, const char* key,
                         const glm::vec3& defaultValue,
                         const std::string& prefix) {
    const nlohmann::json::const_iterator value = object.find(key);
    if (value == object.end()) {
        return defaultValue;
    }
    if (!value->is_array() || value->size() != 3) {
        throw std::runtime_error(prefix + key +
                                 " must be an array of three numbers");
    }
    for (std::size_t component = 0; component < 3; ++component) {
        if (!(*value)[component].is_number()) {
            throw std::runtime_error(prefix + key +
                                     " must contain only numbers");
        }
    }
    return glm::vec3((*value)[0].get<float>(),
                     (*value)[1].get<float>(),
                     (*value)[2].get<float>());
}

glm::vec2 ReadObjectVec2(const nlohmann::json& object, const char* key,
                         const glm::vec2& defaultValue,
                         const std::string& prefix) {
    const nlohmann::json::const_iterator value = object.find(key);
    if (value == object.end()) {
        return defaultValue;
    }
    if (!value->is_array() || value->size() != 2) {
        throw std::runtime_error(prefix + key +
                                 " must be an array of two numbers");
    }
    for (std::size_t component = 0; component < 2; ++component) {
        if (!(*value)[component].is_number()) {
            throw std::runtime_error(prefix + key +
                                     " must contain only numbers");
        }
    }
    return glm::vec2((*value)[0].get<float>(),
                     (*value)[1].get<float>());
}

bool ReadBool(const nlohmann::json& root, const char* key,
              bool defaultValue) {
    const nlohmann::json::const_iterator value = root.find(key);
    if (value == root.end()) {
        return defaultValue;
    }
    if (!value->is_boolean()) {
        throw std::runtime_error(std::string(key) + " must be a boolean");
    }
    return value->get<bool>();
}

float ReadFloat(const nlohmann::json& object, const char* key,
                float defaultValue, const std::string& prefix) {
    const nlohmann::json::const_iterator value = object.find(key);
    if (value == object.end()) {
        return defaultValue;
    }
    if (!value->is_number()) {
        throw std::runtime_error(prefix + key + " must be a number");
    }
    return value->get<float>();
}

bool ReadObjectBool(const nlohmann::json& object, const char* key,
                    bool defaultValue, const std::string& prefix) {
    const nlohmann::json::const_iterator value = object.find(key);
    if (value == object.end()) {
        return defaultValue;
    }
    if (!value->is_boolean()) {
        throw std::runtime_error(prefix + key + " must be a boolean");
    }
    return value->get<bool>();
}

std::uint32_t ReadUint(const nlohmann::json& object, const char* key,
                       std::uint32_t defaultValue,
                       const std::string& prefix) {
    const nlohmann::json::const_iterator value = object.find(key);
    if (value == object.end()) {
        return defaultValue;
    }
    if (!value->is_number_integer()) {
        throw std::runtime_error(prefix + key + " must be an integer");
    }
    const std::int64_t parsed = value->get<std::int64_t>();
    if (parsed < 0 ||
        parsed > static_cast<std::int64_t>(
            std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error(prefix + key +
                                 " is outside the uint32 range");
    }
    return static_cast<std::uint32_t>(parsed);
}

} // namespace

bool Load(const std::string& path, Data& output, std::string& error) {
    output = Data();
    error.clear();

    try {
        std::ifstream stream(path.c_str());
        if (!stream) {
            throw std::runtime_error("Cannot open scene file: " + path);
        }

        nlohmann::json root;
        stream >> root;
        if (!root.is_object()) {
            throw std::runtime_error("Scene root must be a JSON object");
        }

        output.showGround = ReadBool(root, "showGround", true);
        output.showGrass = ReadBool(root, "showGrass", true);
        output.grass.enabled = output.showGrass;

        const std::string baseDirectory = ParentDirectory(path);

        const nlohmann::json::const_iterator grass = root.find("grass");
        if (grass != root.end()) {
            if (!grass->is_object()) {
                throw std::runtime_error("grass must be an object");
            }
            output.grass.enabled = ReadObjectBool(
                *grass, "enabled", output.showGrass, "grass.");
            output.grass.count = ReadUint(
                *grass, "count", output.grass.count, "grass.");
            output.grass.seed = ReadUint(
                *grass, "seed", output.grass.seed, "grass.");
            output.grass.center = ReadObjectVec3(
                *grass, "center", output.grass.center, "grass.");
            output.grass.extent = ReadObjectVec2(
                *grass, "extent", output.grass.extent, "grass.");
            output.grass.position = ReadObjectVec3(
                *grass, "position", output.grass.position, "grass.");
            output.grass.rotationDegrees = ReadObjectVec3(
                *grass, "rotation", output.grass.rotationDegrees,
                "grass.");
            output.grass.scale = ReadObjectVec3(
                *grass, "scale", output.grass.scale, "grass.");
            output.grass.minHeight = ReadFloat(
                *grass, "minHeight", output.grass.minHeight, "grass.");
            output.grass.maxHeight = ReadFloat(
                *grass, "maxHeight", output.grass.maxHeight, "grass.");
            output.grass.minWidth = ReadFloat(
                *grass, "minWidth", output.grass.minWidth, "grass.");
            output.grass.maxWidth = ReadFloat(
                *grass, "maxWidth", output.grass.maxWidth, "grass.");
            output.grass.minStiffness = ReadFloat(
                *grass, "minStiffness", output.grass.minStiffness,
                "grass.");
            output.grass.maxStiffness = ReadFloat(
                *grass, "maxStiffness", output.grass.maxStiffness,
                "grass.");
            output.grass.bottomColor = ReadObjectVec3(
                *grass, "bottomColor", output.grass.bottomColor,
                "grass.");
            output.grass.topColor = ReadObjectVec3(
                *grass, "topColor", output.grass.topColor, "grass.");
            output.grass.rimColor = ReadObjectVec3(
                *grass, "rimColor", output.grass.rimColor, "grass.");
            output.grass.flowersEnabled = ReadObjectBool(
                *grass, "flowersEnabled", output.grass.flowersEnabled,
                "grass.");
            output.grass.flowerDensity = ReadFloat(
                *grass, "flowerDensity", output.grass.flowerDensity,
                "grass.");
            output.grass.flowerHeightScale = ReadFloat(
                *grass, "flowerHeightScale",
                output.grass.flowerHeightScale, "grass.");

            const nlohmann::json::const_iterator positions =
                grass->find("positions");
            if (positions != grass->end()) {
                if (!positions->is_string() ||
                    positions->get<std::string>().empty()) {
                    throw std::runtime_error(
                        "grass.positions must be a non-empty string");
                }
                output.grass.positionsPath = ResolvePath(
                    baseDirectory, positions->get<std::string>());
            }

            if (output.grass.enabled && output.grass.count == 0) {
                throw std::runtime_error(
                    "grass.count must be greater than zero");
            }
            if (output.grass.extent.x <= 0.0f ||
                output.grass.extent.y <= 0.0f) {
                throw std::runtime_error(
                    "grass.extent components must be positive");
            }
            if (output.grass.minHeight <= 0.0f ||
                output.grass.maxHeight < output.grass.minHeight ||
                output.grass.minWidth <= 0.0f ||
                output.grass.maxWidth < output.grass.minWidth ||
                output.grass.minStiffness <= 0.0f ||
                output.grass.maxStiffness < output.grass.minStiffness) {
                throw std::runtime_error(
                    "grass height, width, and stiffness ranges are invalid");
            }
            if (output.grass.scale.x == 0.0f ||
                output.grass.scale.y == 0.0f ||
                output.grass.scale.z == 0.0f) {
                throw std::runtime_error(
                    "grass.scale components must be non-zero");
            }
            if (output.grass.flowerDensity < 0.0f ||
                output.grass.flowerDensity > 1.0f ||
                output.grass.flowerHeightScale <= 0.0f) {
                throw std::runtime_error(
                    "grass flowerDensity must be in [0, 1] and "
                    "flowerHeightScale must be positive");
            }
            if (glm::any(glm::lessThan(output.grass.bottomColor,
                                       glm::vec3(0.0f))) ||
                glm::any(glm::greaterThan(output.grass.bottomColor,
                                          glm::vec3(1.0f))) ||
                glm::any(glm::lessThan(output.grass.topColor,
                                       glm::vec3(0.0f))) ||
                glm::any(glm::greaterThan(output.grass.topColor,
                                          glm::vec3(1.0f))) ||
                glm::any(glm::lessThan(output.grass.rimColor,
                                       glm::vec3(0.0f))) ||
                glm::any(glm::greaterThan(output.grass.rimColor,
                                          glm::vec3(1.0f)))) {
                throw std::runtime_error(
                    "grass colors must be within [0, 1]");
            }
        }
        output.showGrass = output.grass.enabled;

        const nlohmann::json::const_iterator camera = root.find("camera");
        if (camera != root.end()) {
            if (!camera->is_object()) {
                throw std::runtime_error("camera must be an object");
            }
            output.camera.position = ReadObjectVec3(
                *camera, "position", output.camera.position, "camera.");
            output.camera.target = ReadObjectVec3(
                *camera, "target", output.camera.target, "camera.");
            output.camera.fieldOfViewDegrees = ReadFloat(
                *camera, "fov", output.camera.fieldOfViewDegrees,
                "camera.");
            output.camera.nearPlane = ReadFloat(
                *camera, "near", output.camera.nearPlane, "camera.");
            output.camera.farPlane = ReadFloat(
                *camera, "far", output.camera.farPlane, "camera.");
            if (output.camera.fieldOfViewDegrees <= 1.0f ||
                output.camera.fieldOfViewDegrees >= 179.0f ||
                output.camera.nearPlane <= 0.0f ||
                output.camera.farPlane <= output.camera.nearPlane ||
                glm::length(output.camera.position - output.camera.target) <
                    0.001f) {
                throw std::runtime_error("camera parameters are invalid");
            }
            output.camera.enabled = true;
        }

        const nlohmann::json::const_iterator environment =
            root.find("environment");
        if (environment != root.end()) {
            if (!environment->is_object()) {
                throw std::runtime_error("environment must be an object");
            }
            const nlohmann::json::const_iterator environmentPath =
                environment->find("path");
            if (environmentPath == environment->end() ||
                !environmentPath->is_string() ||
                environmentPath->get<std::string>().empty()) {
                throw std::runtime_error(
                    "environment.path must be a non-empty string");
            }
            output.environment.path = ResolvePath(
                baseDirectory, environmentPath->get<std::string>());
            output.environment.backgroundIntensity = ReadFloat(
                *environment, "backgroundIntensity", 1.0f,
                "environment.");
            output.environment.lightingIntensity = ReadFloat(
                *environment, "lightingIntensity", 1.0f,
                "environment.");
            output.environment.directLightingIntensity = ReadFloat(
                *environment, "directLightingIntensity", 1.0f,
                "environment.");
            output.environment.rotationDegrees = ReadFloat(
                *environment, "rotation", 0.0f, "environment.");
            output.environment.visible = ReadObjectBool(
                *environment, "visible", true, "environment.");
            output.environment.enabled = true;
            if (output.environment.backgroundIntensity < 0.0f ||
                output.environment.lightingIntensity < 0.0f ||
                output.environment.directLightingIntensity < 0.0f) {
                throw std::runtime_error(
                    "environment intensities must be non-negative");
            }
        }

        const nlohmann::json::const_iterator models = root.find("models");
        if (models == root.end() || !models->is_array()) {
            throw std::runtime_error("Scene must contain a models array");
        }

        output.models.reserve(models->size());
        for (std::size_t i = 0; i < models->size(); ++i) {
            const nlohmann::json& model = (*models)[i];
            if (!model.is_object()) {
                throw std::runtime_error(
                    "models[" + std::to_string(i) + "] must be an object");
            }
            const nlohmann::json::const_iterator modelPath =
                model.find("path");
            if (modelPath == model.end() || !modelPath->is_string() ||
                modelPath->get<std::string>().empty()) {
                throw std::runtime_error(
                    "models[" + std::to_string(i) +
                    "].path must be a non-empty string");
            }

            ModelEntry entry;
            entry.path = ResolvePath(
                baseDirectory, modelPath->get<std::string>());
            entry.position = ReadVec3(
                model, "position", glm::vec3(0.0f), i);
            entry.rotationDegrees = ReadVec3(
                model, "rotation", glm::vec3(0.0f), i);
            entry.scale = ReadVec3(model, "scale", glm::vec3(1.0f), i);
            if (entry.scale.x == 0.0f || entry.scale.y == 0.0f ||
                entry.scale.z == 0.0f) {
                throw std::runtime_error(
                    "models[" + std::to_string(i) +
                    "].scale components must be non-zero");
            }
            output.models.push_back(entry);
        }

        if (output.models.empty()) {
            throw std::runtime_error("Scene models array cannot be empty");
        }
    } catch (const std::exception& exception) {
        error = exception.what();
        output = Data();
        return false;
    }

    return true;
}

} // namespace SceneConfig
