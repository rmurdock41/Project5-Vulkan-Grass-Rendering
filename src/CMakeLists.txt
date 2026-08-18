file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

file(GLOB IMAGES
    ${CMAKE_CURRENT_SOURCE_DIR}/images/*.jpg
    ${CMAKE_CURRENT_SOURCE_DIR}/images/*.png
    ${CMAKE_CURRENT_SOURCE_DIR}/images/*.bmp
)

foreach(IMAGE ${IMAGES})
    get_filename_component(fname ${IMAGE} NAME)
    configure_file(${IMAGE} ${CMAKE_SOURCE_DIR}/bin/images/${fname} COPYONLY)
endforeach()

file(GLOB MODEL_ASSETS
    ${CMAKE_CURRENT_SOURCE_DIR}/models/*.glb
    ${CMAKE_CURRENT_SOURCE_DIR}/models/*.gltf
)

file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/models)
foreach(MODEL_ASSET ${MODEL_ASSETS})
    get_filename_component(fname ${MODEL_ASSET} NAME)
    configure_file(${MODEL_ASSET} ${CMAKE_SOURCE_DIR}/bin/models/${fname} COPYONLY)
endforeach()

file(GLOB SCENE_ASSETS ${CMAKE_CURRENT_SOURCE_DIR}/scenes/*.json)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/scenes)
foreach(SCENE_ASSET ${SCENE_ASSETS})
    get_filename_component(fname ${SCENE_ASSET} NAME)
    configure_file(${SCENE_ASSET} ${CMAKE_SOURCE_DIR}/bin/scenes/${fname} COPYONLY)
endforeach()

file(GLOB GRASS_ASSETS ${CMAKE_CURRENT_SOURCE_DIR}/grass/*.grass)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/grass)
foreach(GRASS_ASSET ${GRASS_ASSETS})
    get_filename_component(fname ${GRASS_ASSET} NAME)
    configure_file(${GRASS_ASSET} ${CMAKE_SOURCE_DIR}/bin/grass/${fname} COPYONLY)
endforeach()

file(GLOB ENVIRONMENT_ASSETS
    ${CMAKE_CURRENT_SOURCE_DIR}/environments/*.hdr
    ${CMAKE_CURRENT_SOURCE_DIR}/environments/*.exr
    ${CMAKE_CURRENT_SOURCE_DIR}/environments/*.png
)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/environments)
foreach(ENVIRONMENT_ASSET ${ENVIRONMENT_ASSETS})
    get_filename_component(fname ${ENVIRONMENT_ASSET} NAME)
    configure_file(
        ${ENVIRONMENT_ASSET}
        ${CMAKE_SOURCE_DIR}/bin/environments/${fname}
        COPYONLY
    )
endforeach()

file(GLOB_RECURSE SHADER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/*.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/*.geom
    ${CMAKE_CURRENT_SOURCE_DIR}/*.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.tese
    ${CMAKE_CURRENT_SOURCE_DIR}/*.tesc
)

source_group("Shaders" FILES ${SHADER_SOURCES})

if(WIN32)
    add_executable(vulkan_grass_rendering WIN32 ${SOURCES} ${SHADER_SOURCES})
    target_link_libraries(vulkan_grass_rendering ${WINLIBS})
else(WIN32)
    add_executable(vulkan_grass_rendering ${SOURCES})
    target_link_libraries(vulkan_grass_rendering ${CMAKE_THREAD_LIBS_INIT})
endif(WIN32)

foreach(SHADER_SOURCE ${SHADER_SOURCES})
    set(SHADER_DIR ${CMAKE_SOURCE_DIR}/bin/shaders) 
    if(WIN32)
        get_filename_component(fname ${SHADER_SOURCE} NAME)
        add_custom_target(${fname}.spv
            COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_DIR} && 
            $ENV{VK_SDK_PATH}/Bin/glslangValidator.exe -V ${SHADER_SOURCE} -o ${SHADER_DIR}/${fname}.spv -g
            SOURCES ${SHADER_SOURCE}
        )
        ExternalTarget("Shaders" ${fname}.spv)
        add_dependencies(vulkan_grass_rendering ${fname}.spv)
    endif(WIN32)

    # TODO: Build shaders on not windows
endforeach()

target_link_libraries(vulkan_grass_rendering ${ASSIMP_LIBRARIES} Vulkan::Vulkan glfw)
target_include_directories(vulkan_grass_rendering PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/external
  ${GLM_INCLUDE_DIR}
  ${STB_INCLUDE_DIR}
)

# CPU-only glTF validation utility. It deliberately does not create Vulkan
# resources, so loader bugs can be diagnosed independently from the renderer.
add_executable(gltf_inspect
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/gltf_inspect.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GltfLoader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TinyGltfImpl.cpp
)
target_include_directories(gltf_inspect PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/external
    ${GLM_INCLUDE_DIR}
)
InternalTarget("Tools" gltf_inspect)

add_executable(frustum_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/frustum_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Frustum.cpp
)
target_include_directories(frustum_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/external
    ${GLM_INCLUDE_DIR}
)
InternalTarget("Tools" frustum_test)

InternalTarget("" vulkan_grass_rendering)
