#include "test.cuh"

int simpleTest(int argc, char** argv){
    //if(argc != 8){
        //std::cout<<"not enough entries"<<std::endl;
        //return -1;
    //}
    size_t nc = 3;
    size_t w = 7; size_t h = 7;
    size_t m = 5; size_t n = 5;
    //size_t w = std::stoi(argv[1]); size_t h = std::stoi(argv[2]);
    //size_t m = std::stoi(argv[4]); size_t n = std::stoi(argv[5]);
    size_t outSizeX = m + w - 1;  size_t outSizeY = n + h - 1;
    size_t inSize = w * h * nc;
    size_t outSize = outSizeX * outSizeY * nc;
    size_t outSizeXdc = m;    size_t outSizeYdc = n;
    size_t outSizedc = outSizeXdc * outSizeYdc * nc;
    float* imgIn = new float[inSize];
    float* imgOut = new float[outSize];
    float* imgUpConv =  new float[outSize];
    float* kernel = new float[m*n];
    float* imgOutDownConv = new float[outSizeXdc * outSizeYdc * nc];
    float* imgBuffer = new float[inSize];
    for(int i=0; i<inSize; ++i)
        imgIn[i] = 1.0f;
    for(int i=0; i<m*n; ++i)
        kernel[i] = 1.0f;

    //CPU commands
    /*padImgCPU(imgOut, imgIn, w, h, nc, m, n);*/
    /*std::cout<<"padding done"<<std::endl;*/
    /*computeUpConvolutionCPU(imgUpConv, imgIn, kernel, w, h, nc, m, n);*/

    //GPU commnds
    float* d_imgIn = NULL;
    float* d_imgOut = NULL;
    float* d_kernel = NULL;
    float* d_imgUpConv = NULL;
    float* d_imgDownConv = NULL;
    float* d_imgInBuffer = NULL;
    cudaMalloc(&d_imgIn, inSize*sizeof(float));
    cudaMalloc(&d_imgInBuffer, inSize*sizeof(float));
    cudaMalloc(&d_imgOut, outSize*sizeof(float));
    cudaMalloc(&d_kernel, m*n*sizeof(float));
    cudaMalloc(&d_imgUpConv, outSize*sizeof(float));
    cudaMalloc(&d_imgDownConv, outSizedc*sizeof(float));
    cudaMemcpy(d_imgIn, imgIn, inSize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, m*n*sizeof(float), cudaMemcpyHostToDevice);
    padImgGlobalMemCuda(d_imgOut, d_imgIn, w, h, nc, m, n);
    computeUpConvolutionGlobalMemCuda(d_imgUpConv, d_imgIn, d_kernel, w, h, nc, m, n);
    computeImageConvolution(d_imgDownConv, m, n,
                            d_imgIn, d_imgInBuffer, 
                            w, h, 
                            d_imgOut, outSizeX, outSizeY, 
                            nc); 
    cudaMemcpy(imgOut, d_imgOut, outSize*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(imgUpConv, d_imgUpConv, outSize*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(imgOutDownConv, d_imgDownConv, outSizedc*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(imgBuffer, d_imgInBuffer, inSize*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout<<"Input Image"<<std::endl;
    for(int c=0; c<nc; ++c){
        std::cout<<"Channel no :  "<<c<<std::endl;
        for(int j=0; j<h; ++j){
            for(int i=0; i<w; ++i){
                std::cout<<imgIn[i + (j*w) + (c*w*h)]<<"   ";
            }
            std::cout<<std::endl;
        }
    }

    std::cout<<"Output Image after padding"<<std::endl;
    for(int c=0; c<nc; ++c){
        std::cout<<"Channel no :  "<<c<<std::endl;
        for(int j=0; j<outSizeY; ++j){
            for(int i=0; i<outSizeX; ++i){
                std::cout<<imgOut[i + (j*outSizeX) + (c*outSizeX*outSizeY)]<<"   ";
            }
            std::cout<<std::endl;
        }
    }

    std::cout<<"After Up convolution"<<std::endl;
    for(int c=0; c<nc; ++c){
        std::cout<<"Channel no :  "<<c<<std::endl;
        for(int j=0; j<outSizeY; ++j){
            for(int i=0; i<outSizeX; ++i){
                std::cout<<imgUpConv[i + (j*outSizeX) + (c*outSizeX*outSizeY)]<<"   ";
            }
            std::cout<<std::endl;
        }
    }

    std::cout << "After Ravil's Down Conv" << std::endl;
    for(int j=0; j<n; ++j){
        for(int i=0; i<m; ++i){
            std::cout<<imgOutDownConv[i + (j*m)]<<"   ";
        }
        std::cout<<std::endl;
    }

    std::cout<<"Buffer"<<std::endl;
    for(int c=0; c<nc; ++c){
        std::cout<<"Channel no :  "<<c<<std::endl;
        for(int j=0; j<h; ++j){
            for(int i=0; i<w; ++i){
                std::cout<<imgBuffer[i + (j*w) + (c*w*h)]<<"   ";
            }
            std::cout<<std::endl;
        }
    }
    //Free Cuda Memory
    cudaFree(d_imgIn);
    cudaFree(d_imgOut);
    cudaFree(d_kernel);
    cudaFree(d_imgUpConv);
    cudaFree(d_imgDownConv);
    cudaFree(d_imgInBuffer);

    //Free Host Memory
    delete[] imgIn;
    delete[] imgOut;
    delete[] imgUpConv;
    delete[] kernel;
    delete[] imgOutDownConv;
    delete[] imgBuffer;
    
}
