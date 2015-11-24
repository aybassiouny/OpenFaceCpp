cmake_minimum_required(VERSION 2.8)
project( OpenFaceCpp )
find_package( OpenCV REQUIRED )

include(/home/aybassiouny/Desktop/OpenFaceCpp/Lib/dlib-18.18/dlib/cmake/)
#find_package(unittest-cpp REQUIRED)
include_directories(/home/aybassiouny/Desktop/OpenFaceCpp/Lib/unittest-cpp/)
link_directories(/home/aybassiouny/Desktop/OpenFaceCpp/Lib/unittest-cpp/builds/)
add_executable( OpenFaceCpp NativeDLib.cpp )

SET(GCC_COVERAGE_COMPILE_FLAGS "-fdiagnostics-color -std=c++11")
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

TARGET_LINK_LIBRARIES(OpenFaceCpp dlib)
TARGET_LINK_LIBRARIES(OpenFaceCpp UnitTest++)
target_link_libraries( OpenFaceCpp ${OpenCV_LIBS} )