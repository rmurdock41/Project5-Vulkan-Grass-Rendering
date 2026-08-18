#include "GltfLoader.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: gltf_inspect <model.glb|model.gltf>\n";
        return 2;
    }

    Gltf::SceneData scene;
    std::string errors;
    std::string warnings;
    if (!Gltf::Loader::Load(argv[1], scene, errors, warnings)) {
        if (!warnings.empty()) {
            std::cerr << "Warnings:\n" << warnings;
        }
        std::cerr << "Failed to load glTF:\n" << errors;
        return 1;
    }

    if (!warnings.empty()) {
        std::cerr << "Warnings:\n" << warnings;
    }
    Gltf::Loader::PrintSummary(scene, std::cout);
    return 0;
}
