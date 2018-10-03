#include "normalise.cuh"

__global__
void divideByNormKernel(float* kernel, const int m,
        const int n, const float norm1){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    int idy = threadIdx.y + blockDim.y*blockIdx.y;
    int id = idx + idy*m;
    if(idx < m && idy < n){
        kernel[id] /= norm1;
    }
}

void normaliseGlobalMemCuda(float* kernel, const int m, const int n){
    if(!kernel){
        std::cout<< "Kernel not allocated"<<std::endl;
        return;
    }
    // allocate block and grid size
    dim3 block(32, 8, 1);
    dim3 grid = computeGrid2D(block, w + m - 1, h + n - 1);
    
    float norm1 = 0.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSasum(handle, m*n, kernel, 1, &norm1);
    //calling cuda kernel
    divideByNormKernel <<<grid,block>>> (kernel, m, n, norm1);
}
