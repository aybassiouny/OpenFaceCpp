cmake_minimum_required(VERSION 2.8)
include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.12.50.tar.gz"
    SHA1 "51b632f2221ae585045add7a9223343ae51ae843"
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

SET(GCC_COVERAGE_COMPILE_FLAGS "-DDLIB_PNG_SUPPORT -DDLIB_JPEG_SUPPORT -DDLIB_USE_BLAS -DDLIB_USE_LAPACK -DNDEBUG -DDLIB_DISABLE_ASSERTS")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
IF(UNIX)
    add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})
ENDIF(UNIX)

find_package( OpenCV REQUIRED )

include(${CMAKE_CURRENT_SOURCE_DIR}/Lib/dlib-18.18/dlib/cmake/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Lib/tinyxml2)

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

add_executable(OpenFaceCpp ${SOURCES} ${HEADERS})
TARGET_LINK_LIBRARIES(OpenFaceCpp dlib svmlib UnitTest++ lapack blas ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Boost_SYSTEM_LIBRARY} BoostProcess::boost_process)

#add_executable(run_tests src/run_tests.cpp src/NativeDLib.cpp src/TorchWrap.cpp )
#TARGET_LINK_LIBRARIES(run_tests dlib svmlib UnitTest++ lapack blas ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Boost_SYSTEM_LIBRARY} BoostProcess::boost_process)
#add_executable( OpenFaceCppApp simClassification.cpp svm.cpp NativeDLib.cpp TorchWrap.cpp )
#find_package(unittest-cpp REQUIRED)

#TARGET_LINK_LIBRARIES(OpenFaceCpp dlib svmlib UnitTest++ ${OpenCV_LIBS})
