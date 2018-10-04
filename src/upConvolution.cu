#include "upConvolution.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"


__global__
void computeUpConvolutionGlobalMemKernel(float* imgOut, const float* imgIn,
        const float* kernel, int w, int h, int nc, int padX, int padY){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int idy = threadIdx.y + blockIdx.y*blockDim.y;
    int id, tempX, tempY;
    int outSizeX = w + padX + padX;
    int outSizeY = h + padY + padY;
    
    if(idx < outSizeX && idy <outSizeY){
        for(int c = 0; c < nc; ++c){
            id = idx + (idy*outSizeX) + (c*outSizeX*outSizeY);
            imgOut[id] = 0.0;
            for(int kernelY = -1*padY; kernelY <= padY; ++kernelY){
                tempY = idy + kernelY - padY;
                for(int kernelX = -1*padX; kernelX <= padX; ++kernelX){
                    tempX = idx + kernelX - padX;
                    if(tempX >= 0 && tempX < w && tempY >= 0 && tempY < h){
                        imgOut[id] += (imgIn[tempX + (tempY*w) + (c*w*h)])
                            * kernel[(padX+kernelX) + ((padY+kernelY)*(padX+padX+1))];
                    }
                }
            }
        }
    }

    /*for (int channel = 0; channel < nc; ++channel) {*/

        /*int offset = w * h * channel;*/
        /*int idx = threadIdx.x + blockIdx.x * blockDim.x;*/
        /*int idy = threadIdx.y + blockIdx.y * blockDim.y;*/

        /*for (int x = idx; x < w; x += blockDim.x * gridDim.x) {*/
            /*for (int y = idy; y < h; y += blockDim.y * gridDim.y) {*/

                /*int id = getIndex(x, y + 2 * padY + 1, w + 2 * padX + 1) + offset; */
                /*float center = imgIn[id];*/
                /*imgOut[id] = center;*/
            /*}*/
        /*}*/
    /*}*/
}


void initialiseKernel(float *kernel, int m, int n){
    const float x = 1.0f/(m*n);
    for(int i=0; i<m*n; ++i)
        kernel[i] = x;
}


void computeUpConvolutionCPU(float *imgOut, const float *imgIn, 
        const float *kernel, int w, int h, int nc, int m, int n){
    size_t outSizeX = m+w-1;  size_t outSizeY = n+h-1;
    int padX = floor(m/2.0);  int  padY = floor(n/2.0);
    size_t inSize = w*h*nc;
    size_t outSize = (m+w-1) * (n+h-1) * nc;
    int id, idx, idy = 0;
    for(int c = 0; c < nc; ++c){
        for(int j = 0; j < outSizeY; ++j){
            for(int i = 0; i < outSizeX; ++i){
                id = i + (j*outSizeX) + (c*outSizeX*outSizeY);
                imgOut[id] = 0.0;
                for(int kernelY = -1*padY; kernelY <= padY; ++kernelY){
                    idy = j+kernelY-padY;
                    for(int kernelX = -1*padX; kernelX <= padX; ++kernelX){
                        idx = i+kernelX-padX;
                        if(idx >= 0 && idx < w && idy >=0 && idy < h){
                            imgOut[id] += (imgIn[idx + (idy*w) + (c*w*h)]
                                    * kernel[(padX+kernelX) + ((padY+kernelY)*m)]);
                        }
                    }
                }
            }
        }
    }
}

void computeUpConvolutionGlobalMemCuda(float *imgOut, const float *imgIn,
        const float *kernel, int w, int h, int nc, int m, int n){
    if(!imgIn){
        std::cout<< " input not allocated"<<std::endl;
        return;
    }
    if(!imgOut){
        std::cout << "output not allocated" << std::endl;         
        return;   
    }
    if (!imgOut || !imgIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
                            
    }

    // allocate block and grid size
    dim3 block(32, 8, 1);
    int padX = (int) m/2.0f;
    int padY = (int) n/2.0f;
    dim3 grid = computeGrid2D(block, w + m - 1, h + n - 1);

    //calling cuda kernel
    computeUpConvolutionGlobalMemKernel <<<grid,block>>> (imgOut, imgIn, kernel,
                                            w, h, nc, padX, padY);
    CUDA_CHECK;
}
