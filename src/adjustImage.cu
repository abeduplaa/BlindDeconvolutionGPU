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


    dim3 block(32, 1, 1);
    dim3 grid = computeGrid1D(block, mk * nk);
    
    rotateKernel_180_Kernel<<<grid, block>>>(d_kernel_rot_180, d_kernel, mk * nk);
    CUDA_CHECK;
} 


__global__
void stackDataKernel(float *d_data, 
                     const int num_elements,
                     const int nc) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    for (int channel = 1; channel < nc; ++channel) {
        for (int x = idx; x < num_elements; x += blockDim.x * gridDim.x) {
            int offset = num_elements * channel;
            d_data[idx + offset] = d_data[idx]; 
        }
    }
}

void stackData(float *d_data, 
               const int w, 
               const int h,
               const int nc) {


    dim3 block(32, 1, 1);
    dim3 grid = computeGrid1D(block, w * h);
    
    stackDataKernel<<<grid, block>>>(d_data, w * h, nc);
    CUDA_CHECK;
} 
