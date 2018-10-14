#include "cudnnUpConvolution.cuh"

size_t createDescriptorsUp(cudnnTensorDescriptor_t& input_descriptor,
                           cudnnFilterDescriptor_t& kernel_descriptor,
                           cudnnConvolutionDescriptor_t& convolution_descriptor,
                           cudnnTensorDescriptor_t& output_descriptor,
                           cudnnConvolutionFwdAlgo_t& convolution_algorithm,
                           cudnnHandle_t& cudnn,
                           const int inputX,
                           const int inputY,
                           const int kernelX,
                           const int kernelY,
                           const int nc){

    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnSetTensor4dDescriptor(input_descriptor,
                               /*format=*/CUDNN_TENSOR_NCHW,
                               /*dataType=*/CUDNN_DATA_FLOAT,
                               /*batch_size=*/1,
                               /*channels=*/1,
                               /*image_height=*/inputY,
                               /*image_width=*/inputX);
  
    cudnnCreateFilterDescriptor(&kernel_descriptor);
    cudnnSetFilter4dDescriptor(kernel_descriptor,
                               /*dataType=*/CUDNN_DATA_FLOAT,
                               /*format=*/CUDNN_TENSOR_NCHW,
                               /*out_channels=*/1,
                               /*in_channels=*/1,
                               /*kernel_height=*/kernelY,
                               /*kernel_width=*/kernelX);

    cudnnCreateConvolutionDescriptor(&convolution_descriptor);
    cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                    /*pad_height=*/kernelY - 1,
                                    /*pad_width=*/kernelX - 1,
                                    /*vertical_stride=*/1,
                                    /*horizontal_stride=*/1,
                                    /*dilation_height=*/1,
                                    /*dilation_width=*/1,
                                    /*mode=*/CUDNN_CROSS_CORRELATION,
                                    /*computeType=*/CUDNN_DATA_FLOAT);

    int batch_size{0}, channels{0}, height{0}, width{0};
    cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                          input_descriptor,
                                          kernel_descriptor,
                                          &batch_size,
                                          &channels,
                                          &height,
                                          &width);

    std::cerr << "Output Image: " << height << " x " << width << " x " << channels << std::endl;

    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnSetTensor4dDescriptor(output_descriptor,
                               /*format=*/CUDNN_TENSOR_NHWC,
                               /*dataType=*/CUDNN_DATA_FLOAT,
                               /*batch_size=*/1,
                               /*channels=*/channels,
                               /*image_height=*/height,
                               /*image_width=*/width);

    cudnnGetConvolutionForwardAlgorithm(cudnn,
                                        input_descriptor,
                                        kernel_descriptor,
                                        convolution_descriptor,
                                        output_descriptor,
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                        /*memoryLimitInBytes=*/0,
                                        &convolution_algorithm);

    size_t workspace_bytes{0};
    cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                            input_descriptor,
                                            kernel_descriptor,
                                            convolution_descriptor,
                                            output_descriptor,
                                            convolution_algorithm,
                                            &workspace_bytes);
    std::cerr << "Workspace bytes: " << workspace_bytes << std::endl;
    std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB" << std::endl;
    assert(workspace_bytes > 0);


    return workspace_bytes;
}

void callConvolutionUp(float* d_output,
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
                       size_t workspace_bytes){

    const float alpha = 1.0f, beta = 0.0f;
    for(int c = 0; c < nc; ++c){
        cudnnConvolutionForward(cudnn,
                                &alpha,
                                input_descriptor,
                                &d_input[inputX * inputY * c],
                                kernel_descriptor,
                                d_kernel,
                                convolution_descriptor,
                                convolution_algorithm,
                                d_workspace,
                                workspace_bytes,
                                &beta,
                                output_descriptor,
                                &d_output[outputX * outputY * c]);
        cudaThreadSynchronize();
    }
}
