cmake_minimum_required(VERSION 3.15...3.26)
project(py3dti VERSION ${SKBUILD_PROJECT_VERSION})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


include_directories("${PROJECT_SOURCE_DIR}/3dti_AudioToolkit/3dti_Toolkit")
include_directories("${PROJECT_SOURCE_DIR}/3dti_AudioToolkit/3dti_ResourceManager")
include_directories("${PROJECT_SOURCE_DIR}/3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/boost_circular_buffer")
include_directories("${PROJECT_SOURCE_DIR}/3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/cereal/include")
include_directories("${PROJECT_SOURCE_DIR}/3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src")
include_directories("${PROJECT_SOURCE_DIR}/3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include")

link_directories("${PROJECT_SOURCE_DIR}/3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/lib")
if(WIN32)
    set(PLATFORM "win")
elseif(APPLE)
    set(PLATFORM "macos")
else()
    set(PLATFORM "linux")
endif()
link_directories("${PROJECT_SOURCE_DIR}/3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/lib/${PLATFORM}")

link_libraries(sofa curl hdf5 hdf5_hl netcdf netcdf_c++4 z)

file(GLOB_RECURSE SOURCES
3dti_AudioToolkit/3dti_Toolkit/BinauralSpatializer/*.cpp
3dti_AudioToolkit/3dti_Toolkit/Common/*.cpp
3dti_AudioToolkit/3dti_ResourceManager/HRTF/*.cpp
3dti_AudioToolkit/3dti_ResourceManager/ILD/*.cpp
3dti_AudioToolkit/3dti_ResourceManager/BRIR/*.cpp
)

file(GLOB_RECURSE HEADERS
3dti_AudioToolkit/3dti_Toolkit/BinauralSpatializer/*.h
3dti_AudioToolkit/3dti_Toolkit/Common/*.h
3dti_AudioToolkit/3dti_ResourceManager/HRTF/*.h
3dti_AudioToolkit/3dti_ResourceManager/ILD/*.h
3dti_AudioToolkit/3dti_ResourceManager/BRIR/*.h
)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
python_add_library(${PROJECT_NAME} MODULE py3dti.cpp ${SOURCES} ${HEADERS} WITH_SOABI)
target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::headers)
target_compile_definitions(${PROJECT_NAME} PRIVATE VERSION_INFO=${PROJECT_VERSION})
install(TARGETS ${PROJECT_NAME} DESTINATION .)
