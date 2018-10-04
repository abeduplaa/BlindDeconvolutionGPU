#include <iostream>
// #include <algorithm>

#include "testdownConv.cuh"
#include "downConvolution.cuh"
#include "helper.cuh"
#include "epsilon.cuh"
#include "cublas_v2.h"

void downConvTest(){
    
    // Initialize variables
    size_t nc = 2;
    size_t w = 5; 
    size_t h = 9;
    size_t m = 17; 
    size_t n = 11;
    size_t inSizeX = n+1;
    size_t inSizeY = m+1;
    size_t outSizeX = n-w+1;  
    size_t outSizeY = m-h+1;
    size_t inSize = m*n*nc;
	size_t kernelSize = h*w;
    size_t outSize = outSizeX * outSizeY * nc;


     // Initialize Cublas
     cublasHandle_t handle;
     cublasCreate(&handle);

    float eps = 0;
    float epsCPU = 0;

    // allocate memory for arrays
    float* imgIn = new float[inSize];
    float* dummyGradU = new float[inSize];

    float* kernel = new float[kernelSize];

    float* imgOut = new float[outSize];
    float* imgDownConv =  new float[outSize];
    float* imgDownConvGPU = new float[outSize];
    float* f = new float[outSize];

    for(int i=0; i<inSize; i++)
    {
        imgIn[i] = 3.0f;
        dummyGradU[i] = i * 1.0f;
    }
    
    for(int i=0; i<kernelSize; i++)
    {
        kernel[i] = .35f;
    }    
    
    for(int i=0; i<outSize ; i++)
    {
        f[i] = 3.2f;
        imgOut[i] = 0.0f;
    }
    
    // GPU SECTION

    // Initialize GPU arrays
    float* d_imgIn = NULL;
    float* d_imgOut = NULL;
    float* d_kernel = NULL;
    float* d_imgDownConv = NULL;
    float* d_dummyGradU = NULL;

    // Allocate memory on the GPU:
    cudaMalloc(&d_imgIn, inSize*sizeof(float));
    cudaMalloc(&d_imgOut, outSize*sizeof(float));
    cudaMalloc(&d_kernel, kernelSize*sizeof(float));
    cudaMalloc(&d_imgDownConv, outSize*sizeof(float));
    cudaMalloc(&d_dummyGradU, outSize*sizeof(float));

    
    // Copy arrays from CPU to GPU:
    cudaMemcpy(d_imgIn, imgIn, inSize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSize*sizeof(float), cudaMemcpyHostToDevice);
    
    // Compute downconvolution:
    computeDownConvolutionGlobalMemCuda(d_imgDownConv, d_imgIn, d_kernel, n, m, nc, w, h);

    // Subtract using CUBLAS:
    subtractArraysCUBLAS(handle, d_imgDownConv, f, -1.0f, outSize);

    // Calculate epsilon using CUBLAS
    eps = computeEpsilonCuda(handle, d_imgDownConv, d_dummyGradU, outSize, 5e-3)

    // Copy results back to CPU for visualization:
    cudaMemcpy(imgDownConvGPU, d_imgDownConv, outSize*sizeof(float), cudaMemcpyDeviceToHost);

    // CPU commands

    computeDownConvolutionCPU(imgDownConv, imgIn, kernel, w, h, nc, m, n);

    epsCPU = computeEpsilon(imgIn, dummyGradU, inSize, 0.005);

    subtractArrays(imgOut,imgDownConv, f, outSize);

    
    std::cout<<"Input Image"<<std::endl;
    for(int c=0; c<nc; ++c){
        std::cout<<"Channel no :  "<<c<<std::endl;
        for(int j=0; j<m; ++j){
            for(int i=0; i<n; ++i){
                std::cout<<imgIn[i + (j*w) + (c*w*h)]<<"   ";
            }
            std::cout<<std::endl;
        }
    }

    std::cout<<"After CPU Down convolution"<<std::endl;
    for(int c=0; c<nc; ++c){
        std::cout<<"Channel no :  "<<c<<std::endl;
        for(int j=0; j<outSizeY; ++j){
            for(int i=0; i<outSizeX; ++i){
                std::cout<<imgDownConv[i + (j*outSizeX) + (c*outSizeX*outSizeY)]<<"   ";
            }
            std::cout<<std::endl;
        }
    }

    std::cout<<"After CPU subtraction"<< "\n";
    for(int c=0; c<nc; ++c){
        std::cout<<"Channel no :  "<<c<<std::endl;
        for(int j=0; j<outSizeY; ++j){
            for(int i=0; i<outSizeX; ++i){
                std::cout<<imgOut[i + (j*outSizeX) + (c*outSizeX*outSizeY)]<<"   ";
            }
            std::cout<<std::endl;
        }
    }

    std::cout<<"GPU Deconvolution and Subtraction"<< "\n";
    for(int c=0; c<nc; ++c){
        std::cout<<"Channel no :  "<<c<<std::endl;
        for(int j=0; j<outSizeY; ++j){
            for(int i=0; i<outSizeX; ++i){
                std::cout<<imgDownConvGPU[i + (j*outSizeX) + (c*outSizeX*outSizeY)]<<"   ";
            }
            std::cout<<std::endl;
        }
    }

    std::cout<< "\n" << "GPU epsilon value: " << eps << "\n";
    
    std::cout<< "\n" << "CPU epsilon value: " << epsCPU << "\n";

    //Free Cuda Memory
    cudaFree(d_imgIn);
    cudaFree(d_imgOut);
    cudaFree(d_kernel);
    cudaFree(d_imgDownConv);
    cudaFree(d_dummyGradU);


    //Free Host Memory
    delete[] imgIn;
    delete[] imgOut;
    delete[] imgDownConv;
    delete[] kernel;
    delete[] dummyGradU;
    delete[] f;
    delete[] imgDownConvGPU;
    
    cublasDestroy(handle);
}
