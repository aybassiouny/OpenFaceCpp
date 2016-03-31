cmake_minimum_required(VERSION 2.8)
project( OpenFaceCpp )

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)
find_package(Boost COMPONENTS system REQUIRED) 
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS} /home/aybassiouny/libs/boost-process)
  LINK_DIRECTORIES ( ${Boost_LIBRARY_DIRS} )
endif()


SET(GCC_COVERAGE_COMPILE_FLAGS "-fdiagnostics-color -std=c++11 -O3 -mavx -msse2 -DDLIB_PNG_SUPPORT -DDLIB_JPEG_SUPPORT -DDLIB_USE_BLAS -DDLIB_USE_LAPACK -DNDEBUG -DDLIB_DISABLE_ASSERTS")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

find_package( OpenCV REQUIRED )

include(${CMAKE_CURRENT_SOURCE_DIR}/Lib/dlib-18.18/dlib/cmake/)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Lib/ ${CMAKE_CURRENT_SOURCE_DIR}/Lib/unittest-cpp/)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/Lib ${CMAKE_CURRENT_SOURCE_DIR}/Lib/unittest-cpp/builds/)

add_executable(OpenFaceCpp src/simClassification.cpp src/NativeDLib.cpp src/TorchWrap.cpp )
TARGET_LINK_LIBRARIES(OpenFaceCpp dlib svmlib UnitTest++ lapack blas ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Boost_SYSTEM_LIBRARY})

#add_executable( OpenFaceCppApp simClassification.cpp svm.cpp NativeDLib.cpp TorchWrap.cpp )
#find_package(unittest-cpp REQUIRED)






#TARGET_LINK_LIBRARIES(OpenFaceCpp dlib svmlib UnitTest++ ${OpenCV_LIBS})
