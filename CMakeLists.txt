cmake_minimum_required(VERSION 3.21...4.0)

# vcpkg setup
if(NOT CMAKE_TOOLCHAIN_FILE AND DEFINED ENV{VCPKG_ROOT})
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()
# Static libs; static CRT on Windows to match CMAKE_MSVC_RUNTIME_LIBRARY below
if(NOT VCPKG_TARGET_TRIPLET)
  if(WIN32)
    set(VCPKG_TARGET_TRIPLET "x64-windows-static" CACHE STRING "")
  endif()
endif()

# project setup
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

# headers
include_directories("${PROJECT_SOURCE_DIR}/3dti_AudioToolkit/3dti_Toolkit")
include_directories("${PROJECT_SOURCE_DIR}/3dti_AudioToolkit/3dti_ResourceManager")
include_directories("${PROJECT_SOURCE_DIR}/3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/boost_circular_buffer")
include_directories("${PROJECT_SOURCE_DIR}/3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/cereal/include")
include_directories("${PROJECT_SOURCE_DIR}/3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src")

file(GLOB_RECURSE HEADERS
3dti_AudioToolkit/3dti_Toolkit/BinauralSpatializer/*.h
3dti_AudioToolkit/3dti_Toolkit/Common/*.h
3dti_AudioToolkit/3dti_ResourceManager/HRTF/*.h
3dti_AudioToolkit/3dti_ResourceManager/ILD/*.h
3dti_AudioToolkit/3dti_ResourceManager/BRIR/*.h
)

# sources
set(SOFA_SOURCES
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src/SOFAAPI.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src/SOFAAttributes.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src/SOFACoordinates.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src/SOFADate.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src/SOFAEmitter.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src/SOFAExceptions.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src/SOFAFile.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src/SOFAHelper.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src/SOFAListener.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src/SOFANcFile.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src/SOFAPoint3.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src/SOFAPosition.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src/SOFAReceiver.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src/SOFASimpleFreeFieldHRIR.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src/SOFASimpleFreeFieldSOS.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src/SOFASimpleHeadphoneIR.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src/SOFAMultiSpeakerBRIR.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src/SOFASingleRoomDRIR.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src/SOFAGeneralTF.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src/SOFAGeneralFIR.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src/SOFAGeneralFIRE.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src/SOFASource.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src/SOFAString.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src/SOFAUnits.cpp
)

file(GLOB_RECURSE SOURCES
3dti_AudioToolkit/3dti_Toolkit/BinauralSpatializer/*.cpp
3dti_AudioToolkit/3dti_Toolkit/Common/*.cpp
3dti_AudioToolkit/3dti_ResourceManager/HRTF/*.cpp
3dti_AudioToolkit/3dti_ResourceManager/ILD/*.cpp
3dti_AudioToolkit/3dti_ResourceManager/BRIR/*.cpp
)

# dependencies
find_package(netCDFCxx CONFIG REQUIRED)   # provides netCDF::netcdf-cxx4 (and netCDF::netcdf, HDF5, zlib, curl transitively)
link_libraries(netCDF::netcdf-cxx4)
if(WIN32)
  link_libraries(winmm.lib)
endif()

# python module configuration
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
python_add_library(${PROJECT_NAME} MODULE py3dti.cpp ${SOURCES} ${SOFA_SOURCES} ${HEADERS} WITH_SOABI)
target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::headers)
target_compile_definitions(${PROJECT_NAME} PRIVATE VERSION_INFO=${PROJECT_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_VISIBILITY_PRESET hidden)
install(TARGETS ${PROJECT_NAME} DESTINATION .)
