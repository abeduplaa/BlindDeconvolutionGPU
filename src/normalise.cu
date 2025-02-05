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
    dim3 grid = computeGrid2D(block, m, n);
    
    float norm1 = 0.0;
    //TODO: pass handle as parameter
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    cublasSasum(handle, m*n, kernel, 1, &norm1);
    cublasDestroy(handle);
    //calling cuda kernel
    divideByNormKernel <<<grid,block>>> (kernel, m, n, norm1);
}
