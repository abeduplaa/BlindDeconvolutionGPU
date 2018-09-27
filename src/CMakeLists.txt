cmake_minimum_required(VERSION 2.8)

project(deblur)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3; -std=c++11; -arch=sm_50; -Xptxas -v)

#add_definitions(-DDEBUG)
file(GLOB SRCS "./*.cu")

#include_directories(${CMAKE_SOURCE_DIR}/headers)

cuda_add_executable(deblur ${SRCS})
target_link_libraries(deblur -lm ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})
# Add CUBLAS to target
