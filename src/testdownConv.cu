#include <iostream>

#include "testdownConv.cuh"
#include "downConvolution.cuh"
#include "helper.cuh"
#include "epsilon.cuh"

void downConvTest(){
    
    size_t nc = 1;
    size_t w = 3; 
    size_t h = 3;
    size_t m = 11; 
    size_t n = 11;
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
    
    computeDownConvolutionCPU(imgDownConv, imgIn, kernel, w, h, nc, m, n);

    eps = computeEpsilonU(imgIn, dummyGradU, inSize);

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
        for(int j=0; j<outSizeY; ++j){
            for(int i=0; i<outSizeX; ++i){
                std::cout<<imgOut[i + (j*outSizeX) + (c*outSizeX*outSizeY)]<<"   ";
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
