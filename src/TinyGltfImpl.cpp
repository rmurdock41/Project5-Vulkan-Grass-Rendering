#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_IMPLEMENTATION
#include <tinygltf/tiny_gltf.h>

// TinyGLTF's constructor references its default image callbacks even when
// stb decoding is disabled. The project installs a custom callback before
// loading, so these fallbacks only satisfy the symbols and provide a clear
// error if they are ever reached accidentally.
namespace tinygltf {

bool LoadImageData(Image*, int, std::string* err, std::string*, int, int,
                   const unsigned char*, int, void*) {
    if (err) {
        *err += "TinyGLTF default image decoder is disabled.\n";
    }
    return false;
}

bool WriteImageData(const std::string*, const std::string*, const Image*, bool,
                    const FsCallbacks*, const URICallbacks*, std::string*,
                    void*) {
    return false;
}

} // namespace tinygltf
