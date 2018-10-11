#include "divergence.cuh"
#include "helper.cuh"

#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include "helper.cuh"
#include <stdio.h>


__global__
void computeGradientsKernel(float *dx_fw, float *dy_fw,
                            float *dx_bw, float  *dy_bw,
                            float *dx_mixed, float *dy_mixed, 
                            const float *imgIn, const int w, const int h, const int nc) {

    for (int channel = 0; channel < nc; ++channel) {

        int offset = w * h * channel;
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int idy = threadIdx.y + blockIdx.y * blockDim.y;

        // TODO: flip x and y loops
        for (int x = idx; x < w; x += blockDim.x * gridDim.x) {
            for (int y = idy; y < h; y += blockDim.y * gridDim.y) {
                float center = imgIn[getIndex(x, y, w) + offset];

                float right = (x < w) ? imgIn[getIndex(x + 1, y, w) + offset] : center;  
                float left = (x > 0) ? imgIn[getIndex(x - 1, y, w) + offset] : center;  

                float top = (y < h) 
                          ? imgIn[getIndex(x, y + 1, w) + offset] : center;  
                float down = (y > 0) 
                           ? imgIn[getIndex(x, y - 1, w) + offset] : center;  

                float top_left = ((x > 0) && (y < h - 1)) 
                                   ? imgIn[getIndex(x - 1, y + 1, w) + offset] : left;

                float down_right = ((x < w - 1) && (y > 0)) 
                                    ? imgIn[getIndex(x + 1, y - 1, w) + offset] : down;
                /*float top_right = ((x < w) && (y < h)) */
                               /*? imgIn[getIndex(x + 1, y + 1, w) + offset] : top;*/

                /*float down_left = ((x > 0) && (y > 0)) */
                                 /*? imgIn[getIndex(x - 1, y - 1, w) + offset] : left;*/

                dx_fw[getIndex(x, y, w) + offset] = right - center;
                dy_fw[getIndex(x, y, w) + offset] = top - center;

                dx_bw[getIndex(x, y, w) + offset] = center - left;
                dy_bw[getIndex(x, y, w) + offset] = center - down;

                dx_mixed[getIndex(x, y, w) + offset] = down_right - down;
                dy_mixed[getIndex(x, y, w) + offset] = top_left - left;
                /*dx_mixed[getIndex(x, y, w) + offset] = top_right - top;*/
                /*dy_mixed[getIndex(x, y, w) + offset] = down_left - left;*/
            }
        }
    }
}


__device__
float computeDuffusivityDevice(const float a, const float  b, const float eps) {
    float length = sqrt(a * a + b * b);
    return length > eps ? length : eps; 
}


__global__
void computeDivergenceKernel(float *div, 
                             const float *dx_fw, const float *dy_fw,
                             const float *dx_bw, const float *dy_bw,
                             const float *dx_mixed, const float *dy_mixed, 
                             const float *imgIn, const int w, const int h, const int nc,
                             const float eps) {

    if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
        printf("EPS on gpu: %f\n", eps);

    for (int channel = 0; channel < nc; ++channel) {

        int offset = w * h * channel;
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int idy = threadIdx.y + blockIdx.y * blockDim.y;

        for (int x = idx; x < w; x += blockDim.x * gridDim.x) {
            for (int y = idy; y < h; y += blockDim.y * gridDim.y) {
                int index = getIndex(x, y, w) + offset; 

                float norm_1 = computeDuffusivityDevice(dx_fw[index], dy_fw[index], eps); 
                float sum = dx_fw[index] + dy_fw[index];
                div[index] = sum / norm_1;

                /*div[index] = (dx_fw[index] + dy_fw[index]);*/

                /*div[index] = ((dx_fw[index] + dy_fw[index]) */
                                        /*/ computeDuffusivity(dx_fw[index], dy_fw[index], eps)) */
                           /*- dx_bw[index] / computeDuffusivity(dx_bw[index], dy_mixed[index], eps)*/
                           /*- dy_bw[index] / computeDuffusivity(dx_mixed[index], dy_bw[index], eps); */
            }
        }
    }
}


void computeDiffOperatorsCuda(float *d_div, 
                              float *d_dx_fw, float *d_dy_fw,
                              float *d_dx_bw, float *d_dy_bw,
                              float *d_dx_mixed, float *d_dy_mixed, 
                              const float *d_imgIn, const int w, const int h, const int nc,
                              const float eps) {
    dim3 block(32, 2, 1);
    dim3 grid = computeGrid2D(block, w, h);
    
    computeGradientsKernel<<<grid, block>>>(d_dx_fw, d_dy_fw,
                                            d_dx_bw, d_dy_bw,
                                            d_dx_mixed, d_dy_mixed, 
                                            d_imgIn, w, h, nc);

    cudaDeviceSynchronize();
    computeDivergenceKernel<<<grid, block>>>(d_div, 
                                             d_dx_fw, d_dy_fw,
                                             d_dx_bw, d_dy_bw,
                                             d_dx_mixed, d_dy_mixed, 
                                             d_imgIn, w, h, nc,
                                             eps);
    CUDA_CHECK;
}

