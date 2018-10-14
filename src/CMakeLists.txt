cmake_minimum_required(VERSION 2.8)

project(deblur)


find_package(CUDA REQUIRED)
find_package(OpenCV COMPONENTS
    core highgui imgproc video
    REQUIRED
)
if(opencv_core_found)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()


option(DEBUG_MODE "Option description" OFF)
if(DEBUG_MODE)
    add_definitions(-DDEBUG)
endif(DEBUG_MODE)


include_directories(${PYTHON_INCLUDE_DIRS})


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3; -std=c++11; --use_fast_math; -arch=sm_50; -Xptxas -v)


file(GLOB SRCS "./*.cu")


find_package(PythonLibs 3.5 REQUIRED)
include_directories(${CMAKE_SOURCE_DIR})


cuda_add_executable(deblur ${SRCS})


target_link_libraries(deblur -lm -lcudnn ${PYTHON_LIBRARIES} ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})
