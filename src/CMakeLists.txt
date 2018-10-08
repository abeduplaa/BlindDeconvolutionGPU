cmake_minimum_required(VERSION 2.8)

project(deblur)

find_package(CUDA REQUIRED)
#find_package(OpenCV REQUIRED)
# OpenCV
find_package(OpenCV COMPONENTS
    core highgui imgproc video
    REQUIRED
)
if(opencv_core_found)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()
find_package(PythonLibs 3.5 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})



set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3; -std=c++11; -arch=sm_50; -Xptxas -v)

#add_definitions(-DDEBUG)
file(GLOB SRCS "./*.cu")

include_directories(${CMAKE_SOURCE_DIR})
message( ${CMAKE_SOURCE_DIR} )
message( ${OpenCV_INCLUDE_DIRS} )
cuda_add_executable(deblur ${SRCS})
target_link_libraries(deblur -lm ${PYTHON_LIBRARIES} ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})
# Add CUBLAS to target
