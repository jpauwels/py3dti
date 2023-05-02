cmake_minimum_required(VERSION 3.15...3.26)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15")
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
project(py3dti VERSION ${SKBUILD_PROJECT_VERSION})


include_directories("${PROJECT_SOURCE_DIR}/3dti_AudioToolkit/3dti_Toolkit")
include_directories("${PROJECT_SOURCE_DIR}/3dti_AudioToolkit/3dti_ResourceManager")
include_directories("${PROJECT_SOURCE_DIR}/3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/boost_circular_buffer")
include_directories("${PROJECT_SOURCE_DIR}/3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/cereal/include")
include_directories("${PROJECT_SOURCE_DIR}/3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/src")

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
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncAtt.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncByte.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncChar.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncCheck.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncCompoundType.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncDim.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncDouble.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncEnumType.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncException.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncFile.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncFloat.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncGroup.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncGroupAtt.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncInt.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncInt64.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncOpaqueType.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncShort.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncString.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncType.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncUbyte.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncUint.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncUint64.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncUshort.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncVar.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncVarAtt.cpp
3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies/include/ncVlenType.cpp
)

set(DEPENDENCIES_DIR "${PROJECT_SOURCE_DIR}/3dti_AudioToolkit/3dti_ResourceManager/third_party_libraries/sofacoustics/libsofa/dependencies")

if(WIN32)
	add_library(Hdf5 SHARED IMPORTED)
	add_library(Hdf5_HL SHARED IMPORTED)
	add_library(NetCdf SHARED IMPORTED)
	add_library(Curl SHARED IMPORTED)
	add_library(ZLib SHARED IMPORTED)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
         set(SUFFIX "_x64")
         set(SUBDIR "x64")
    else()
        set(SUBDIR "i386")
    endif()
    set_target_properties(Hdf5 PROPERTIES
		IMPORTED_LOCATION "${DEPENDENCIES_DIR}/lib/win/${SUBDIR}/hdf5.dll"
		IMPORTED_IMPLIB "${DEPENDENCIES_DIR}/lib/win/libhdf5${SUFFIX}.lib"
		INTERFACE_INCLUDE_DIRECTORIES "${DEPENDENCIES_DIR}/include"
	)
    set_target_properties(Hdf5_HL PROPERTIES
		IMPORTED_LOCATION "${DEPENDENCIES_DIR}/lib/win/${SUBDIR}/hdf5_hl.dll"
		IMPORTED_IMPLIB "${DEPENDENCIES_DIR}/lib/win/libhdf5_hl${SUFFIX}.lib"
		INTERFACE_INCLUDE_DIRECTORIES "${DEPENDENCIES_DIR}/include"
	)
    set_target_properties(NetCdf PROPERTIES
		IMPORTED_LOCATION "${DEPENDENCIES_DIR}/lib/win/${SUBDIR}/netcdf.dll"
		IMPORTED_IMPLIB "${DEPENDENCIES_DIR}/lib/win/netcdf${SUFFIX}.lib"
		INTERFACE_INCLUDE_DIRECTORIES "${DEPENDENCIES_DIR}/include"
	)
    set_target_properties(Curl PROPERTIES
		IMPORTED_LOCATION "${DEPENDENCIES_DIR}/lib/win/${SUBDIR}/libcurl.dll"
		IMPORTED_IMPLIB "${DEPENDENCIES_DIR}/lib/win/libcurl_imp${SUFFIX}.lib"
		INTERFACE_INCLUDE_DIRECTORIES "${DEPENDENCIES_DIR}/include"
	)
    set_target_properties(ZLib PROPERTIES
		IMPORTED_LOCATION "${DEPENDENCIES_DIR}/lib/win/${SUBDIR}/zlib1.dll"
		IMPORTED_IMPLIB "${DEPENDENCIES_DIR}/lib/win/zlib${SUFFIX}.lib"
		INTERFACE_INCLUDE_DIRECTORIES "${DEPENDENCIES_DIR}/include"
	)
	link_libraries(winmm.lib)
else()
	add_library(Hdf5 STATIC IMPORTED)
	add_library(Hdf5_HL STATIC IMPORTED)
	add_library(NetCdf STATIC IMPORTED)
	add_library(Curl STATIC IMPORTED)
	add_library(ZLib STATIC IMPORTED)
    if(APPLE)
        set(PLATFORM "macos")
    else()
        set(PLATFORM "linux")
    endif()
    set_target_properties(Hdf5 PROPERTIES
		IMPORTED_LOCATION "${DEPENDENCIES_DIR}/lib/${PLATFORM}/libhdf5.a"
		INTERFACE_INCLUDE_DIRECTORIES "${DEPENDENCIES_DIR}/include"
	)
    set_target_properties(Hdf5_HL PROPERTIES
		IMPORTED_LOCATION "${DEPENDENCIES_DIR}/lib/${PLATFORM}/libhdf5_hl.a"
		INTERFACE_INCLUDE_DIRECTORIES "${DEPENDENCIES_DIR}/include"
	)
    set_target_properties(NetCdf PROPERTIES
		IMPORTED_LOCATION "${DEPENDENCIES_DIR}/lib/${PLATFORM}/libnetcdf.a"
		INTERFACE_INCLUDE_DIRECTORIES "${DEPENDENCIES_DIR}/include"
	)
    set_target_properties(Curl PROPERTIES
		IMPORTED_LOCATION "${DEPENDENCIES_DIR}/lib/${PLATFORM}/libcurl.a"
		INTERFACE_INCLUDE_DIRECTORIES "${DEPENDENCIES_DIR}/include"
	)
    set_target_properties(ZLib PROPERTIES
		IMPORTED_LOCATION "${DEPENDENCIES_DIR}/lib/${PLATFORM}/libz.a"
		INTERFACE_INCLUDE_DIRECTORIES "${DEPENDENCIES_DIR}/include"
	)
endif()
link_libraries(Hdf5 Hdf5_HL NetCdf Curl ZLib)

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
python_add_library(${PROJECT_NAME} MODULE py3dti.cpp ${SOURCES} ${SOFA_SOURCES} ${HEADERS} WITH_SOABI)
target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::headers)
target_compile_definitions(${PROJECT_NAME} PRIVATE VERSION_INFO=${PROJECT_VERSION})
install(TARGETS ${PROJECT_NAME} DESTINATION .)

install(FILES $<TARGET_RUNTIME_DLLS:${PROJECT_NAME}> DESTINATION .) # includes Python dll
