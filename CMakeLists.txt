cmake_minimum_required(VERSION 2.8)
include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.20.78.tar.gz"
    SHA1 "4c50cfb0dbf489aa5b15a0df65671efa744800f8"
)

project( OpenFaceCpp )
hunter_add_package(OpenCV)
hunter_add_package(Boost COMPONENTS system regex)
hunter_add_package(BoostProcess)
hunter_add_package(PNG)

find_package(Boost COMPONENTS system regex REQUIRED) 

find_package(BoostProcess CONFIG REQUIRED)

SET(GCC_COVERAGE_COMPILE_FLAGS "-DDLIB_PNG_SUPPORT -DDLIB_JPEG_SUPPORT -DNDEBUG -DDLIB_DISABLE_ASSERTS")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
IF(UNIX)
    add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})
ENDIF(UNIX)

find_package( OpenCV REQUIRED )

include(${CMAKE_CURRENT_SOURCE_DIR}/Lib/dlib-18.18/dlib/cmake/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Lib/tinyxml2)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/Lib/tinyxml2/)

link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/Lib)

SET(SOURCES
  src/NativeDLib.cpp
  src/main.cpp
  src/TorchWrap.cpp
)

SET(HEADERS
  src/NativeDLib.h
  src/TorchWrap.h
)

SET(LINK_LIBS
	dlib 
	#lapack 
	#blas 
	${OpenCV_LIBS} 
	#Boost::system
	BoostProcess::boost_process
	tinyxml2
)

add_executable(OpenFaceCpp ${SOURCES} ${HEADERS})
TARGET_LINK_LIBRARIES(OpenFaceCpp ${LINK_LIBS})