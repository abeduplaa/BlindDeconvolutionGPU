#include "divergence.cuh"
#include "helper.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"
#include <stdio.h>

inline __host__ __device__ int getIndex(int i, int j, int width) {
    return i + j * width;
}

__global__
void computeGradientsKernel(float *dx_fw, float *dy_fw,
                            float *dx_bw, float *dy_bw,
                            float *dx_mixed, float *dy_mixed, 
                            const float *imgIn, int w, int h, int nc) {

    for (int channel = 0; channel < nc; ++channel) {

        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int idy = threadIdx.y + blockIdx.y * blockDim.y;

        for (int x = idx; x < w; x += blockDim.x * gridDim.x) {
            for (int y = idy; y < h; y += blockDim.y * gridDim.y) {
                float center = imgIn[getIndex(x, y, w)];

                /*float right = (x > w - 1) ? imgIn[getIndex(x + 1, y, w)] : 0.0f;  */
                /*float left = (x > 0) ? imgIn[getIndex(x - 1, y, w)] : 0.0f;  */

                /*float top = (y > h - 1) ? imgIn[getIndex(x, y + 1, w)] : 0.0f;  */
                /*float down = (y > 0) ? imgIn[getIndex(x, y - 1, w)] : 0.0f;  */

                /*float top_left = ((x > 0) && (y > h - 1)) ? imgIn[getIndex(x - 1, y + 1, w)] : 0.0f;*/
                /*float down_right = ((x > w - 1) && (y > 0)) ? imgIn[getIndex(x + 1, y - 1, w)] : 0.0f;*/

                /*dx_fw[getIndex(x, y, w)] = right - center;*/
                /*dy_fw[getIndex(x, y, w)] = top - center;*/

                /*dx_bw[getIndex(x, y, w)] = center - left;*/
                /*dy_bw[getIndex(x, y, w)] = center - down;*/

                /*dx_mixed[getIndex(x, y, w)] = down_right - down;*/
                /*dy_mixed[getIndex(x, y, w)] = top_left - left;*/
                
                /*printf("%f\n", dx_bw[getIndex(x,y,w)]);*/
                dx_fw[getIndex(x,y,w)] =imgIn[getIndex(x,y,w)];
            }
        }
    }
}



void computeDiffOperatorsCuda(float *d_div, 
                              float *d_dx_fw, float *d_dy_fw,
                              float *d_dx_bw, float *d_dy_bw,
                              float *d_dx_mixed, float *d_dy_mixed, 
                              const float *d_imgIn, size_t w, size_t h, size_t nc) {
    dim3 block(32, 4, 1);
    dim3 grid = computeGrid2D(block, w, h);
    
                /*printf("%u %u %u\n", w, h, nc);*/
    computeGradientsKernel<<<grid, block>>>(d_dx_fw, d_dy_fw,
                                            d_dx_bw, d_dy_bw,
                                            d_dx_mixed, d_dy_mixed, 
                                            d_imgIn, w, h, nc);
    CUDA_CHECK;
}



/*void computeDivergence(float *div, float *dx, float *dy,*/
                       /*const float *imgIn, size_t w, size_t h, size_t nc) {*/
    
    /*// init arrays*/
    /*float init_value = 0.0;*/
    /*for (size_t i = 0; i < (w * h * nc); ++i) */
        /*div[i] = dx[i] = dy[i] = init_value; */


    /*// compute gradient using forward differences*/
    /*for (size_t channel = 0; channel < nc; ++channel) {*/

        /*// update the core of the layer*/
        /*for (size_t y = 0; y < h - 1; ++y) {*/
            /*for (size_t x = 0; x < w - 1; ++x) {*/

                /*int offset = w * h * channel;*/
                /*float right_cell = x < (w - 1) ? imgIn[getIndex(x + 1, y, w) + offset] : 0.0f;*/
                /*float top_cell = y < (h - 1) ? imgIn[getIndex(x, y + 1, w) + offset] : 0.0f;*/

                /*dx[getIndex(x, y, w) + offset] = right_cell - imgIn[getIndex(x, y, w) + offset];*/
                /*y[getIndex(x, y, w) + offset] = top_cell - imgIn[getIndex(x, y, w) + offset];*/
            /*}*/
        /*}*/
    /*}*/

    /*// compute divergence*/
    /*for (size_t channel = 0; channel < nc; ++channel) {*/

        /*// update the core of the layer*/
        /*for (size_t y = 0; y < h; ++y) {*/
            /*for (size_t x = 0; x < w; ++x) {*/
                
                /*int offset = w * h * channel;*/

                /*float left_cell = x > 0 ? dx[getIndex(x - 1, y, w) + offset] : 0.0f;*/
                /*float bottom_cell = y > 0 ? dy[getIndex(x, y - 1, w) + offset] : 0.0f;*/

                /*div[getIndex(x, y, w) + offset] = dx[getIndex(x, y, w) + offset] - left_cell*/
                                                /*+ dy[getIndex(x, y, w) + offset] - bottom_cell;*/
            /*}*/
        /*}*/
    /*}*/
/*}*/


/*__device__ */
/*int getShrMemIndex(int x, int y, int width) {*/
    /*return (x + y * (width + 1));*/
/*}*/


/*__global__*/
/*void computeGradientKernel(float *u, float *v, const float *imgIn, int w, int h, int nc) {*/
    /*// (4.1) compute gradient in x-direction (u) and y-direction (v)*/
    /*extern __shared__ float patch[];*/
    /*for (int channel = 0; channel < nc; ++channel) {*/

        /*int idx = threadIdx.x + blockIdx.x * blockDim.x;*/
        /*int idy = threadIdx.y + blockIdx.y * blockDim.y;*/
        /*int offset = channel * w * h;*/

        /*for (int x = idx; x < w; x += blockDim.x * gridDim.x) {*/
            /*for (int y = idy; y < h; y += blockDim.y * gridDim.y) {*/
                /*int index = x + w * y + offset;*/
                
                /*int shr_mem_index = getShrMemIndex(threadIdx.x, threadIdx.y, blockDim.x);*/
                /*int shr_mem_index_right = getShrMemIndex(threadIdx.x + 1, threadIdx.y, blockDim.x);*/
                /*int shr_mem_index_bottom = getShrMemIndex(threadIdx.x, threadIdx.y + 1, blockDim.x);*/

                /*patch[shr_mem_index] = imgIn[index];*/

                /*// load halo region*/
                /*if (threadIdx.y == (blockDim.y - 1)) {*/
                    /*if (idy == (h - 1)) {*/
                        /*patch[shr_mem_index_bottom] = 0.0;*/
                    /*}*/
                    /*else {*/
                        /*int shifted_index = x + w * (y + 1) + offset;*/
                        /*patch[shr_mem_index_bottom] = imgIn[shifted_index];*/
                    /*}*/
                /*}*/
                
                /*if (threadIdx.x == (blockDim.x - 1)) {*/
                    /*if (idx == (w - 1)) {*/
                        /*patch[shr_mem_index_right] = 0.0;*/
                    /*}*/
                    /*else {*/
                        /*int shifted_index = (x + 1) + w * y + offset;*/
                        /*patch[shr_mem_index_right] = imgIn[shifted_index];*/
                    /*}*/
                /*}*/

                /*__syncthreads();*/

                /*u[index] = patch[shr_mem_index_right] - patch[shr_mem_index];*/
                /*v[index] = patch[shr_mem_index_bottom] - patch[shr_mem_index];*/

            /*}*/
        /*}*/
    /*}*/
/*}*/


/*void computeGradientCuda(float *u, float *v, const float *imgIn, int w, int h, int nc)*/
/*{*/
    /*// calculate block and grid size*/
    /*dim3 block(32, 8, 1);     // (4.1) specify suitable block size*/
    /*dim3 grid = computeGrid2D(block, w, h);*/

    /*// run cuda kernel*/
    /*// (4.1) execute gradient kernel*/

    /*// allocate shared memory including the halo region and run the kernel*/
    /*size_t shr_mem_size = (block.x + 1) * (block.y + 1) * sizeof(float);*/
    /*computeGradientKernel<<<grid, block, shr_mem_size>>>(u, v, imgIn, w, h, nc); CUDA_CHECK;*/

    /*// check for errors*/
    /*// (4.1)*/
/*}*/

/*void computeDivergenceCuda(float *Out, const float *v1, const float *v2, int w, int h, int nc) {*/
    /*// calculate block and grid size*/
    /*dim3 block(32, 4, 1);     // (4.2) specify suitable block size*/
    /*dim3 grid = computeGrid2D(block, w, h);*/

    /*// run cuda kernel*/
    /*// (4.2) execute divergence kernel*/
    /*computeDivergenceKernel<<<grid, block>>>(q, v1, v2, w, h, nc);*/
    /*CUDA_CHECK;    */
    /*// check for errors*/
    /*// (4.2)*/
/*}*/


