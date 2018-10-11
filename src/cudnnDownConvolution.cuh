#ifndef CUDNN_DOWNCONVOLUTION_H
#define CUDNN_DOWNCONVOLUTION_H

#include <iostream>
#include <cudnn.h>
#include <cassert>
#include <cstdlib>
#include "cuda_runtime.h"
#include "helper.cuh"


size_t createDescriptorsdc0(cudnnTensorDescriptor_t& input_descriptor,
                       cudnnFilterDescriptor_t& kernel_descriptor,
                       cudnnConvolutionDescriptor_t& convolution_descriptor,
                       cudnnTensorDescriptor_t& output_descriptor,
                       cudnnConvolutionFwdAlgo_t& convolution_algorithm,
                       cudnnHandle_t& cudnn,
                       const int inputX,
                       const int inputY,
                       const int kernelX,
                       const int kernelY,
                       const int nc);

size_t createDescriptorsdc1(cudnnTensorDescriptor_t& input_descriptor,
                       cudnnFilterDescriptor_t& kernel_descriptor,
                       cudnnConvolutionDescriptor_t& convolution_descriptor,
                       cudnnTensorDescriptor_t& output_descriptor,
                       cudnnConvolutionFwdAlgo_t& convolution_algorithm,
                       cudnnHandle_t& cudnn,
                       const int inputX,
                       const int inputY,
                       const int kernelX,
                       const int kernelY,
                       const int nc);

void callConvolutiondc1(float* d_output, 
                     const float* d_input,
                     const float* d_kernel,
                     cudnnTensorDescriptor_t& input_descriptor,
                     cudnnFilterDescriptor_t& kernel_descriptor,
                     cudnnConvolutionDescriptor_t& convolution_descriptor,
                     cudnnTensorDescriptor_t& output_descriptor,
                     cudnnConvolutionFwdAlgo_t& convolution_algorithm,
                     void* d_workspace,
                     cudnnHandle_t& cudnn,
                     size_t workspace_bytes);

void callConvolutiondc0(float* d_output,
                     const float* d_input,
                     const float* d_kernel,
                     const int inputX,
                     const int inputY,
                     const int kernelX,
                     const int kernelY,
                     const int outputX,
                     const int outputY,
                     const int nc,
                     cudnnTensorDescriptor_t& input_descriptor,
                     cudnnFilterDescriptor_t& kernel_descriptor,
                     cudnnConvolutionDescriptor_t& convolution_descriptor,
                     cudnnTensorDescriptor_t& output_descriptor,
                     cudnnConvolutionFwdAlgo_t& convolution_algorithm,
                     void* d_workspace,
                     cudnnHandle_t& cudnn,
                     size_t workspace_bytes);

void destroyDescriptors(cudnnTensorDescriptor_t& input_descriptor,
                       cudnnFilterDescriptor_t& kernel_descriptor,
                       cudnnConvolutionDescriptor_t& convolution_descriptor,
                       cudnnTensorDescriptor_t& output_descriptor);

#endif
