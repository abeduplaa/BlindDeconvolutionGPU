#include <iostream>
#include <cuda_runtime.h>

#include "downConvolution.cuh"
#include "helper.cuh"


__global__
void computeDownConvolutionGlobalMemKernel(float* imgOut, const float* imgIn,
const float* kernel, const int w, const int h, const int nc, const int m, const int n){
	
	//0. define idxx and idyy in 2d grid
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int idy = threadIdx.y + blockIdx.y*blockDim.y;
	
	//1. define necessary variables
	


	size_t imgOut_h = m - h + 1;
	size_t imgOut_w = n - w + 1;
	int kRadius_w = (w-1)/2;
	int kRadius_h = (h-1)/2;
	int kidx = 0;

	// GPU Parameters
	int i = idx + idy*imgOut_w; // or should it be idx + idy*n?
	int out_idx = 0;
	int in_idx = 0;

	//2. compute downconvolution

	if(idx < imgOut_w && idy < imgOut_h)
	{
		for(int c = 0; c < nc; c++)
		{
			out_idx = i + (c*imgOut_h*imgOut_w);
			
			in_idx = out_idx + (kRadius_h*n) + kRadius_w + (kRadius_w*2)*idy;
			imgOut[out_idx] = 0.0f;
			
			for(int kj = 0; kj < h; kj++)
			{
				for(int ki = 0; ki < w; ki++)
				{
					kidx = in_idx - (kRadius_w - ki) - ( (kRadius_h - kj)*w);
					imgOut[imgOut] += kernel[ki + kj*w] * imgIn[kidx];
				}
			}
		}
	}
}



void computeDownConvolutionCPU(float *imgOut, const float *imgIn, const float *kernel, int w, int h, int nc, int m, int n)
{
	//0. check if kernel dimensions are less than image dimensions ( h>m+1 or also w>n+1)
	if( (h>m+1) || (w>n+1) )
	{
		std::cout<< "WRONG";
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
						imgOut[out_idx] += kernel[ki + kj*w] * imgIn[kidx];
					}
				}
			}
		} 
	}
}

void computeDownConvolutionGlobalMemCuda(float *imgOut, const float *imgIn, const float *kernel, int w, int h, int nc, int m, int n)
{

	// allocate block and grid size
	dim3 block(32, 8, 1);
	dim3 grid = computeGrid2D(block, m - h + 1, n - w + 1);

	//calling cuda kernel
	computeDownConvolutionGlobalMemKernel <<<grid,block>>> (imgOut, imgIn, kernel,
											w, h, nc, m, n);
}



