#include "pad.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"

__global__
void padImgReplicateGlobalMemKernel(float* imgOut, const float* imgIn,
        int w, int h, int nc, int padX, int padY){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int idy = threadIdx.y + blockIdx.y*blockDim.y;
    int id, tempX, tempY;
    int outSizeX = w + padX + padX;
    int outSizeY = h + padY + padY;

    for(int c = 0; c < nc; ++c){
        id = idx + (idy*outSizeX) + (c*outSizeX*outSizeY);
        if(idx < outSizeX && idy < outSizeY){
            tempX = idx - padX;
            tempY = idy - padY;
            if(tempX >= 0 && tempX < w && tempY >= 0 && tempY < h){
                imgOut[id] = imgIn[tempX + (tempY*w) +(c*w*h)];
            }
            else{
                tempX = (tempX >= 0 && tempX < w) ? tempX : (tempX < 0) ? 0 : w - 1;
                tempY = (tempY >= 0 && tempY < h) ? tempY : (tempY < 0) ? 0 : h - 1;
                imgOut[id] = imgIn[tempX + (tempY*w) +(c*w*h)];
            }
        }
    }
}

__global__
void padImgZeroGlobalMemKernel(float* imgOut, const float* imgIn,
        int w, int h, int nc, int padX, int padY){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int idy = threadIdx.y + blockIdx.y*blockDim.y;
    int id, tempX, tempY;
    int outSizeX = w + padX + padX;
    int outSizeY = h + padY + padY;

    for(int c = 0; c < nc; ++c){
        id = idx + (idy*outSizeX) + (c*outSizeX*outSizeY);
        if(idx < outSizeX && idy < outSizeY){
            tempX = idx - padX;
            tempY = idy - padY;
            if(tempX >= 0 && tempX < w && tempY >= 0 && tempY < h){
                imgOut[id] = imgIn[tempX + (tempY*w) +(c*w*h)];
            }
            else{
                imgOut[id] = 0.0f;
            }
        }
    }
}

__global__
void padImgSymmetricGlobalMemKernel(float* imgOut, const float* imgIn,
        int w, int h, int nc, int padX, int padY){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int idy = threadIdx.y + blockIdx.y*blockDim.y;
    int id, tempX, tempY;
    int outSizeX = w + padX + padX;
    int outSizeY = h + padY + padY;

    for(int c = 0; c < nc; ++c){
        id = idx + (idy*outSizeX) + (c*outSizeX*outSizeY);
        if(idx < outSizeX && idy < outSizeY){
            tempX = idx - padX;
            tempY = idy - padY;
            if(tempX >= 0 && tempX < w && tempY >= 0 && tempY < h){
                imgOut[id] = imgIn[tempX + (tempY*w) +(c*w*h)];
            }
            else{
                tempX = (tempX >= 0 && tempX < w) ? tempX : (tempX < 0) ? (-1*tempX) - 1 : 2*w - tempX - 1;
                tempY = (tempY >= 0 && tempY < h) ? tempY : (tempY < 0) ? (-1*tempY) - 1 : 2*h - tempY - 1;
                imgOut[id] = imgIn[tempX + (tempY*w) +(c*w*h)];
            }
        }
    }
}

__global__
void padImgPeriodicGlobalMemKernel(float* imgOut, const float* imgIn,
        int w, int h, int nc, int padX, int padY){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int idy = threadIdx.y + blockIdx.y*blockDim.y;
    int id, tempX, tempY;
    int outSizeX = w + padX + padX;
    int outSizeY = h + padY + padY;

    for(int c = 0; c < nc; ++c){
        id = idx + (idy*outSizeX) + (c*outSizeX*outSizeY);
        if(idx < outSizeX && idy < outSizeY){
            tempX = idx - padX;
            tempY = idy - padY;
            if(tempX >= 0 && tempX < w && tempY >= 0 && tempY < h){
                imgOut[id] = imgIn[tempX + (tempY*w) +(c*w*h)];
            }
            else{
                tempX = (tempX >= 0 && tempX < w) ? tempX : (tempX < 0) ? w + tempX + 1 : tempX - w - 1;
                tempY = (tempY >= 0 && tempY < h) ? tempY : (tempY < 0) ? h + tempY - 1 : tempY - h - 1;
                imgOut[id] = imgIn[tempX + (tempY*w) +(c*w*h)];
            }
        }
    }
}

void padImgCPU(float* imgOut, const float* imgIn,
               size_t w, size_t h, size_t nc,
               size_t m, size_t n){
    size_t padX = floor(m/2.0);
    size_t padY = floor(n/2.0);
    size_t outSizeX = w+padX+padX;
    size_t outSizeY = h+padY+padY;
    size_t totalSize = outSizeX * outSizeY*nc;
    for(int i=0; i<totalSize; ++i)
        imgOut[i] = 0.f;
    for(int c=0; c<nc; ++c){
        for(int j=0; j<h; ++j){
            for(int i=0; i<w; ++i){
                imgOut[padX + i + (j+padY)*outSizeX + c*outSizeX*outSizeY] = 
                    imgIn[i + j*w + c*w*h];
            }
        }
    }
}

void padImgGlobalMemCuda(float *imgOut, const float *imgIn,
                         int w, int h, int nc, int m, int n,
                         BoundaryCondition& bc){
    if(!imgIn){
        std::cout<< " input not allocated"<<std::endl;
        return;
    }
    if(!imgOut){
        std::cout << "output not allocated" << std::endl;         
        return;   
    }

    // allocate block and grid size
    dim3 block(32, 8, 1);
    int padX = (int) floor(m/2.0f);
    int padY = (int) floor(n/2.0f);
    dim3 grid = computeGrid2D(block, w + m - 1, h + n - 1);

    //calling cuda kernel
    if(bc == replicate)
        padImgReplicateGlobalMemKernel <<<grid,block>>> (imgOut, imgIn, w, h, nc, padX, padY);
    else if(bc == zero)
        padImgZeroGlobalMemKernel <<<grid,block>>> (imgOut, imgIn, w, h, nc, padX, padY);
    else if(bc == periodic)
        padImgPeriodicGlobalMemKernel <<<grid,block>>> (imgOut, imgIn, w, h, nc, padX, padY);
    else
        padImgSymmetricGlobalMemKernel <<<grid,block>>> (imgOut, imgIn, w, h, nc, padX, padY);
}

void selectBoundaryCondition(char bc, BoundaryCondition& boundary){
    switch(bc){
        case 'r' :std::cout << "Replicate boundary condition" << std::endl;
                  boundary = replicate;
                  break;
        case 'z' :std::cout << "Zero boundary condition" << std::endl;
                  boundary = zero;
                  break;
        case 'p' :std::cout << "Periodic boundary condition" << std::endl;
                  boundary = periodic;
                  break;
        case 's' :std::cout << "Symmetric boundary condition" << std::endl;
                  boundary = symmetric;
                  break;
        default  :std::cout << "Incoorect entry of boundary condition, default is replicate" << std::endl;
                  boundary = replicate;
    }
}
