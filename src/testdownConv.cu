#include <iostream>

#include "testdownConv.cuh"
#include "downConvolution.cuh"
#include "helper.cuh"
#include "epsilon.cuh"

void downConvTest(){
    
    size_t nc = 1;
    size_t w = 7; 
    size_t h = 11;
    size_t m = 9; 
    size_t n = 9;
    size_t inSizeX = n+1;
    size_t inSizeY = m+1;
    size_t outSizeX = n-w+1;  
    size_t outSizeY = m-h+1;
    size_t inSize = m*n*nc;
	size_t kernelSize = h*w;
    size_t outSize = outSizeX * outSizeY * nc;

    float eps = 0;

    float* imgIn = new float[inSize];
    float* dummyGradU = new float[inSize];

    float* kernel = new float[kernelSize];

    float* imgOut = new float[outSize];
    float* imgDownConv =  new float[outSize];
    float* f = new float[outSize];


    for(int i=0; i<inSize; i++)
    {
        imgIn[i] = 1.0f;
        dummyGradU[i] = i * 1.0f;
    }
    
    for(int i=0; i<kernelSize; i++)
    {
        kernel[i] = 1.0f;
    }    
    
    for(int i=0; i<outSize ; i++)
    {
        f[i] = 3.2f;
    }
    
    //CPU commands
    //padImgCPU(imgOut, imgIn, w, h, nc, m, n);
    /*std::cout<<"padding done"<<std::endl;*/
    
    computeDownConvolutionCPU(imgDownConv, imgIn, kernel, w, h, nc, m, n);

    eps = computeEpsilonU(imgIn, dummyGradU, inSize);

    subtractArrays(imgOut,imgDownConv, f, outSize);

    
    // //GPU commnds
    // float* d_imgIn = NULL;
    // float* d_imgOut = NULL;
    // float* d_kernel = NULL;
    // float* d_imgUpConv = NULL;
    // cudaMalloc(&d_imgIn, inSize*sizeof(float));
    // cudaMalloc(&d_imgOut, outSize*sizeof(float));
    // cudaMalloc(&d_kernel, m*n*sizeof(float));
    // cudaMalloc(&d_imgUpConv, outSize*sizeof(float));
    // cudaMemcpy(d_imgIn, imgIn, inSize*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_kernel, kernel, m*n*sizeof(float), cudaMemcpyHostToDevice);
    // padImgGlobalMemCuda(d_imgOut, d_imgIn, w, h, nc, m, n);
    // computeUpConvolutionGlobalMemCuda(d_imgUpConv, d_imgIn, d_kernel, w, h, nc, m, n);
    // cudaMemcpy(imgOut, d_imgOut, outSize*sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(imgUpConv, d_imgUpConv, outSize*sizeof(float), cudaMemcpyDeviceToHost);

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

    // std::cout<<"Output Image after padding"<<std::endl;
    // for(int c=0; c<nc; ++c){
    //     std::cout<<"Channel no :  "<<c<<std::endl;
    //     for(int j=0; j<outSizeY; ++j){
    //         for(int i=0; i<outSizeX; ++i){
    //             std::cout<<imgOut[i + (j*outSizeX) + (c*outSizeX*outSizeY)]<<"   ";
    //         }
    //         std::cout<<std::endl;
    //     }
    // }

    std::cout<<"After Down convolution"<<std::endl;
    for(int c=0; c<nc; ++c){
        std::cout<<"Channel no :  "<<c<<std::endl;
        for(int j=0; j<outSizeY; ++j){
            for(int i=0; i<outSizeX; ++i){
                std::cout<<imgDownConv[i + (j*outSizeX) + (c*outSizeX*outSizeY)]<<"   ";
            }
            std::cout<<std::endl;
        }
    }

    std::cout<<"After subtraction"<< "\n";
    for(int c=0; c<nc; ++c){
        std::cout<<"Channel no :  "<<c<<std::endl;
        for(int j=0; j<inSizeY; ++j){
            for(int i=0; i<inSizeX; ++i){
                std::cout<<imgOut[i + (j*inSizeX) + (c*inSizeX*inSizeY)]<<"   ";
            }
            std::cout<<std::endl;
        }
    }

    std::cout<<"epsilon value: " << eps << "\n";

    //Free Cuda Memory
    // cudaFree(d_imgIn);
    // cudaFree(d_imgOut);
    // cudaFree(d_kernel);
    // cudaFree(d_imgUpConv);

    //Free Host Memory
    delete[] imgIn;
    delete[] imgOut;
    delete[] imgDownConv;
    delete[] kernel;
    delete[] dummyGradU;
    delete[] f;
    
}
