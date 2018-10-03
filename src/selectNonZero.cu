#include "selectNonZero.cuh"

__global__
void selectNonZeroKernel(float* kkernel){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    int idy = threadIdx.y + blockDim.y*blockIdx.y;
    int id = idx + idy*m;
    if(idx < m && idy < n){
        kernel[id] = (kernel[id] > 0)?kernel[id]:0.0f;
    }
}

void selectNonZeroGlobalMemCuda(float* kernel){
    if(!kernel){
        std::cout<< "Kernel not allocated"<<std::endl;
        return;
    }
    // allocate block and grid size
    dim3 block(32, 8, 1);
    dim3 grid = computeGrid2D(block, w + m - 1, h + n - 1);
    
    //calling cuda kernel
    selectNonZeroKernel <<<grid,block>>> (kernel);
}
