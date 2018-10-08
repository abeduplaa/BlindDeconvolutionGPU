#include <iostream>
#include <cuda_runtime.h>

#include "downConvolution.cuh"
#include "helper.cuh"
#include "../cub-1.8.0/cub/cub.cuh"

__global__
void computeDownConvolutionGlobalMemKernel(float* imgOut, const float* imgIn,
const float* kernel, const int w, const int h, const int nc, const int m, const int n){
	
	//0. define idxx and idyy in 2d grid
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int idy = threadIdx.y + blockIdx.y*blockDim.y;
	
	//1. define necessary variables
	size_t imgOut_w = w - m + 1;
	size_t imgOut_h = h - n + 1;
	int kRadius_m = (m - 1) / 2;
	int kRadius_n = (n - 1) / 2;
	int kidx = 0;

	// GPU Parameters
	int i = idx + idy*imgOut_w;
	int out_idx, out_x, out_y = 0;
	int in_idx = 0;

	//2. compute downconvolution

	//if(idx < imgOut_w && idy < imgOut_h) //why is this incorrect?
	if(idx < imgOut_w && idy < imgOut_h )
	{
		for(int c = 0; c < nc; c++)
		{
            out_idx = i + (c*imgOut_h*imgOut_w);
            out_x = idx + kRadius_m;
            out_y = idy + kRadius_n;
            imgOut[out_idx] = 0.0f;
			
            for(int kj = -kRadius_n; kj <= kRadius_n; kj++)
            {
                for(int ki = -kRadius_m; ki <= kRadius_m; ki++)
                {
                    imgOut[out_idx] += kernel[(ki+kRadius_m)+((kj+kRadius_n)*kRadius_m)]
                        * imgIn[(out_x+ki)+ (out_y+kj)*w + (c*w*h)];
                }
            }
		}
	}
}



void computeDownConvolutionCPU(float *imgOut, const float *imgIn, const float *kernel, int n, int m, int nc, int w, int h)
{
	//0. check if kernel dimensions are less than image dimensions ( h>m+1 or also w>n+1)
	if( (h>m+1) || (w>n+1) )
	{
		std::cout << "\n" << "n: " << n << ", w: " << w << ", m: " << m << ", h: " << h << "\n";
		throw std::invalid_argument("Kernel dimensions are too big!");
	}


	//1. define necessary variables
	int out_idx = 0;
	int in_idx = 0;
	int kidx = 0;
	size_t imgOut_h = m - h + 1;
	size_t imgOut_w = n - w + 1;
	int kRadius_w = (w-1)/2;
	int kRadius_h = (h-1)/2;

	//2. compute downconvolution

	for(int c = 0; c < nc; c++)
	{
		for(int j = 0 ; j < imgOut_h ; j++)
		{
			for(int i = 0 ; i < imgOut_w ; i++)
			{
				out_idx = i + j*imgOut_w + c*imgOut_h*imgOut_w;
				in_idx = out_idx + (kRadius_h*n) + kRadius_w + (kRadius_w*2)*j;
				
				imgOut[out_idx] = 0.0f;
				// std::cout<< "output idx: " << out_idx << ", input idx: " << in_idx << "\n"; 
				
				for(int kj = 0; kj < h; ++kj)
				{
					for(int ki = 0; ki < w; ++ki)
					{
						kidx = in_idx - (kRadius_w - ki) - ( (kRadius_h - kj)*n);
						// std::cout<< "convolution multiplication, k: " << ki+kj*w << ", input: " << kidx << " , value: " << kernel[ki + kj*w] * imgIn[kidx] <<  "\n";
						imgOut[out_idx - c*imgOut_h*imgOut_w] += kernel[ki + kj*w + c*imgOut_h*imgOut_w] * imgIn[kidx];
					}
				}
			}
		} 
	}
}

void computeDownConvolutionGlobalMemCuda(float *imgOut, const float *imgIn, const float *kernel,
        const int w, const int h, const int nc, const int m, const int n)
{

	// allocate block and grid size
	dim3 block(32, 8, 1);
	dim3 grid = computeGrid2D(block, w - m + 1, h - n + 1);

	//calling cuda kernel
	computeDownConvolutionGlobalMemKernel <<<grid,block>>> (imgOut, imgIn, kernel,
											w, h, nc, m, n);
}


// -------------------------------------------------------------------------------------
__global__
void computeImageConvolution(float *imgInBuffer,
                             const float *imgIn,
                             const int w, const int h,
                             const float *imgInPad,
                             const int padw, const int padh,
                             const int delta_x, const int delta_y,
                             const int nc) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    for (int channel = 0; channel < nc; ++channel) {

        int offset = w * h * channel;
        int offset_pad_img = padw * padh * channel;

        for (int y = idy; y < h; y += blockDim.y * gridDim.y) {
            for (int x = idx; x < w; x += blockDim.x * gridDim.x) {

                int relative_x = x + delta_x; 
                int relative_y = y + delta_y; 
                
                int kernel_index = getIndex(x, y, w) + offset;
                int image_index = getIndex(relative_x, relative_y, padw) + offset_pad_img;

                imgInBuffer[kernel_index] = imgIn[kernel_index] * imgInPad[image_index];
            }
        }
    }

}


void computeImageConvolution(float *d_kernel_temp, const int mk, const int nk ,
                             const float *d_imgIn, float *d_imgInBuffer, 
                             const int w, const int h, 
                             const float *d_imgInPad, const int padw, const int padh, 
                             const int nc) {

	// allocate block and grid size
	dim3 block(32, 8, 1);
	dim3 grid = computeGrid2D(block, w, h);
    // determine the size of temp buffer
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // NOTE: When d_temp_storage is NULL, no work is done and
    // the required allocation size is returned in
    // temp_storage_bytes.
    cub::DeviceReduce::Sum<float*, float*>(d_temp_storage,
                                           temp_storage_bytes, 
                                           d_imgInBuffer, 
                                           &d_kernel_temp[0], 
                                           w * h * nc);

    cudaMalloc(&d_temp_storage, temp_storage_bytes); CUDA_CHECK;

    for(int delta_y = 0; delta_y < nk; ++delta_y){
        for(int delta_x = 0; delta_x < mk; ++delta_x){


            int kernel_index = getIndex(delta_x, delta_y, mk);
            computeImageConvolution<<<grid, block>>>(d_imgInBuffer,
                                                     d_imgIn,
                                                     w, h,
                                                     d_imgInPad,
                                                     padw, padh,
                                                     delta_x, delta_y,
                                                     nc);
            
            CUDA_CHECK;
            cudaThreadSynchronize();


            cub::DeviceReduce::Sum<float*, float*>(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_imgInBuffer,
                                                   &d_kernel_temp[kernel_index],
                                                   w * h * nc);
            cudaThreadSynchronize();
        }
    }
    cudaFree(d_temp_storage);  CUDA_CHECK;
}



