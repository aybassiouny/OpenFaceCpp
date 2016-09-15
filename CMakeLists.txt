cmake_minimum_required(VERSION 2.8)
include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.14.1.tar.gz"
    SHA1 "d073cc04d59a01782385799568afce78066a8cec"
)

project( OpenFaceCpp )
hunter_add_package(OpenCV)
hunter_add_package(Boost COMPONENTS system regex)
hunter_add_package(BoostProcess)
hunter_add_package(PNG)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)
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
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Lib/unittest-cpp)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Lib/easylogging)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/Lib/tinyxml2/
    ${CMAKE_CURRENT_SOURCE_DIR}/Lib/unittest-cpp/)

link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/Lib 
    ${CMAKE_CURRENT_SOURCE_DIR}/Lib/unittest-cpp/builds/)

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
	UnitTest++ 
	#lapack 
	#blas 
	${OpenCV_LIBS} 
	#Boost::system
	BoostProcess::boost_process
	tinyxml2
)

add_executable(OpenFaceCpp ${SOURCES} ${HEADERS})
TARGET_LINK_LIBRARIES(OpenFaceCpp ${LINK_LIBS})

#add_executable(run_tests src/run_tests.cpp src/NativeDLib.cpp src/TorchWrap.cpp )
#TARGET_LINK_LIBRARIES(run_tests dlib UnitTest++ lapack blas ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Boost_SYSTEM_LIBRARY} BoostProcess::boost_process)
#add_executable( OpenFaceCppApp simClassification.cpp NativeDLib.cpp TorchWrap.cpp )
#find_package(unittest-cpp REQUIRED)

#TARGET_LINK_LIBRARIES(OpenFaceCpp dlib UnitTest++ ${OpenCV_LIBS})
