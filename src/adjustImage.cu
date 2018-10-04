#include "adjustImage.cuh"

void adjustImageSizeToOdd(cv::Mat& mIn, int& w, int& h, int& nc){
    if(w % 2 == 0)
        --w;
    if(h % 2 == 0)
        --h;
    cv::Rect myRectangle(0, 0, w, h);
    cv::Mat temp =  mIn(myRectangle);
    temp.copyTo(mIn);
}


__global__
void rotateKernel_180_Kernel(float *kernel_rot_180,
                             const float *kernel, 
                             const int N) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    for (int x = idx; x < N; x += blockDim.x * gridDim.x) {
        kernel_rot_180[x] = kernel[N - x - 1];
    }
} 


void rotateKernel_180(float *d_kernel_rot_180,
                      const float *d_kernel, 
                      const int mk, 
                      const int nk) {


    dim3 block(32, 4, 1);
    dim3 grid = computeGrid1D(block, mk * nk);
    
    rotateKernel_180_Kernel<<<grid, block>>>(d_kernel_rot_180, d_kernel, mk * nk);
    CUDA_CHECK;
} 
