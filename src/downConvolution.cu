#include <iostream>
#include <cuda_runtime.h>

#include "downConvolution.cuh"
#include "helper.cuh"


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





